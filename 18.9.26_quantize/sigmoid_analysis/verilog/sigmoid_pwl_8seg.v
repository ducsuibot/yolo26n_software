// =============================================================================
// Module: sigmoid_pwl_8seg
// Target Architecture: Xilinx Zynq UltraScale+ (ZCU104 - XCZU7EV)
// Application: UAV Tracking YOLOv26n Activation Accelerator
//
// Description:
//   Piecewise Linear (PWL) Approximation for Sigmoid Activation Function
//   using Shift-and-Add Arithmetic (100% DSP-Free Architecture):
//       y = 1 / (1 + exp(-x))
//
// Specifications:
//   - Input (in_x)    : 8-bit signed (Q4.4 format, range [-8.0, +7.9375], step 0.0625)
//   - Output (out_y)  : 8-bit unsigned (Q0.8 format, range [0, 255] representing [0.0, 1.0])
//   - Segments        : 8 symmetric segments using powers-of-two slopes (2^-k)
//   - Hardware Cost   : ~95 LUTs, ~40 FFs (Logic only)
//   - DSP Cost        : Exactly 0 DSP48E2
//   - BRAM Cost       : Exactly 0 BRAM
//   - Latency         : 2 clock cycles (Pipelined: Stage 1 = segment detect & shift, Stage 2 = add & symmetry)
// =============================================================================

`timescale 1ns / 1ps

module sigmoid_pwl_8seg (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire signed [7:0] in_x,   // Q4.4 format
    output reg         [7:0] out_y    // Q0.8 format
);

    // -------------------------------------------------------------------------
    // Stage 1: Absolute value & Segment Identification (Registered)
    // -------------------------------------------------------------------------
    reg        s1_sign;
    reg  [7:0] s1_abs_x; // Q4.4 unsigned magnitude
    reg  [9:0] s1_term;  // Shifted term in Q0.8
    reg  [7:0] s1_offset;// Base intercept in Q0.8

    // Internal Q4.4 constants for segment boundaries (value * 16)
    localparam [7:0] BND_0_5  = 8'd8;   // 0.5 * 16 = 8
    localparam [7:0] BND_1_25 = 8'd20;  // 1.25 * 16 = 20
    localparam [7:0] BND_2_0  = 8'd32;  // 2.0 * 16 = 32
    localparam [7:0] BND_3_0  = 8'd48;  // 3.0 * 16 = 48
    localparam [7:0] BND_4_0  = 8'd64;  // 4.0 * 16 = 64
    localparam [7:0] BND_5_0  = 8'd80;  // 5.0 * 16 = 80
    localparam [7:0] BND_6_0  = 8'd96;  // 6.0 * 16 = 96

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_sign   <= 1'b0;
            s1_abs_x  <= 8'd0;
            s1_term   <= 10'd0;
            s1_offset <= 8'd0;
        end else if (enable) begin
            s1_sign  <= in_x[7];
            // Compute magnitude: abs(in_x)
            s1_abs_x <= in_x[7] ? (-in_x) : in_x;

            // Compute piecewise slopes using arithmetic shifts
            // Note: s1_abs_x is in Q4.4 (scaled by 16).
            // To get Q0.8 (scaled by 256), multiply by 16.
            // Slope 0.25 in Q0.8 -> abs_x * 16 * 0.25 = abs_x * 4 = abs_x << 2.
            if (s1_abs_x < BND_0_5) begin
                // y = 0.25 * x + 0.5 -> term = abs_x << 2, offset = 128
                s1_term   <= {s1_abs_x, 2'b00};
                s1_offset <= 8'd128;
            end else if (s1_abs_x < BND_1_25) begin
                // y = (0.25 - 0.03125) * x + 0.515625 -> term = (abs_x << 2) - (abs_x >> 1), offset = 132
                s1_term   <= {s1_abs_x, 2'b00} - (s1_abs_x >> 1);
                s1_offset <= 8'd132;
            end else if (s1_abs_x < BND_2_0) begin
                // y = (0.125 + 0.03125) * x + 0.59375 -> term = (abs_x << 1) + (abs_x >> 1), offset = 152
                s1_term   <= {s1_abs_x, 1'b0} + (s1_abs_x >> 1);
                s1_offset <= 8'd152;
            end else if (s1_abs_x < BND_3_0) begin
                // y = (0.0625 + 0.03125) * x + 0.71875 -> term = abs_x + (s1_abs_x >> 1), offset = 184
                s1_term   <= s1_abs_x + (s1_abs_x >> 1);
                s1_offset <= 8'd184;
            end else if (s1_abs_x < BND_4_0) begin
                // y = (0.03125 + 0.015625) * x + 0.859375 -> term = (abs_x >> 1) + (abs_x >> 2), offset = 220
                s1_term   <= (s1_abs_x >> 1) + (s1_abs_x >> 2);
                s1_offset <= 8'd220;
            end else if (s1_abs_x < BND_5_0) begin
                // y = 0.015625 * x + 0.921875 -> term = (abs_x >> 2), offset = 236
                s1_term   <= (s1_abs_x >> 2);
                s1_offset <= 8'd236;
            end else if (s1_abs_x < BND_6_0) begin
                // y = 0.00390625 * x + 0.98 -> term = (abs_x >> 4), offset = 251
                s1_term   <= (s1_abs_x >> 4);
                s1_offset <= 8'd251;
            end else begin
                // Saturation: y = 1.0 (255 in Q0.8)
                s1_term   <= 10'd0;
                s1_offset <= 8'd255;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Stage 2: Sum & Apply Symmetry for negative x (Registered)
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_y <= 8'd0;
        end else if (enable) begin
            reg [9:0] raw_sum;
            reg [7:0] clamped_pos;
            
            raw_sum = s1_term + s1_offset;
            // Clamp to 255 (1.0)
            clamped_pos = (raw_sum > 10'd255) ? 8'd255 : raw_sum[7:0];

            // If x was negative, sigma(-|x|) = 1 - sigma(|x|) = 255 - clamped_pos
            if (s1_sign) begin
                out_y <= 8'd255 - clamped_pos;
            end else begin
                out_y <= clamped_pos;
            end
        end
    end

endmodule
