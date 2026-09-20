// =============================================================================
// Module: silu_fused_lut_256
// Target Architecture: Xilinx Zynq UltraScale+ (ZCU104 - XCZU7EV)
// Application: UAV Tracking YOLOv26n Activation Accelerator (SiLU Nodes)
//
// Description:
//   FUSED INT8 SiLU (Swish) Activation Table:
//       q_next = clip(round(( (q_Y * Sy) * Sigmoid(q_Y * Sy) ) / Sx_next), -128, 127)
//
//   Eliminates all 4 stages:
//       1. Dequantize to FP32 (Eliminated!)
//       2. Floating-point Sigmoid (Eliminated!)
//       3. Floating-point Multiplication (Eliminated!)
//       4. Quantize to INT8 (Eliminated!)
//
// Specifications:
//   - Input (in_q)    : 8-bit signed INT8 (Direct output of Conv requantization)
//   - Output (out_q)  : 8-bit signed INT8 (Direct input to next Conv layer)
//   - Table Size      : 256 entries x 8 bits = 256 bytes
//   - Resource Cost   : Exactly 32 LUT6 (Distributed ROM) or 0.1% of ZCU104 fabric
//   - DSP Cost        : Exactly 0 DSP48E2
//   - Latency         : 1 clock cycle (Fully pipelined)
// =============================================================================

`timescale 1ns / 1ps

module silu_fused_lut_256 #(
    parameter TABLE_INIT_FILE = "silu_fused_table.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire signed [7:0] in_q,   // INT8 input from Conv block
    output reg  signed [7:0] out_q   // INT8 output to downstream Conv block
);

    // Map signed INT8 [-128..127] to unsigned address [0..255]
    wire [7:0] lut_addr = in_q ^ 8'h80;

    // Distributed ROM synthesis attribute for UltraScale+
    (* rom_style = "distributed" *) reg signed [7:0] rom [0:255];

    initial begin
        $readmemh(TABLE_INIT_FILE, rom);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_q <= 8'sd0;
        end else if (enable) begin
            out_q <= rom[lut_addr];
        end
    end

endmodule
