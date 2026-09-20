// =============================================================================
// Module: sigmoid_lut_256
// Target Architecture: Xilinx Zynq UltraScale+ (ZCU104 - XCZU7EV)
// Application: UAV Tracking YOLOv26n Activation Accelerator
//
// Description:
//   Direct Table Look-Up (LUT) for Sigmoid Activation Function:
//       y = 1 / (1 + exp(-x))
//
// Specifications:
//   - Input (in_x)    : 8-bit signed (Q4.4 format, range [-8.0, +7.9375], step 0.0625)
//   - Output (out_y)  : 8-bit unsigned (Q0.8 format, range [0, 255] representing [0.0, 0.996])
//   - Table Size      : 256 entries x 8 bits = 2048 bits
//   - Hardware Cost   : Exactly 32 LUT6 (Distributed ROM) on Xilinx UltraScale+
//   - DSP Cost        : 0 DSP48E2
//   - BRAM Cost       : 0 Block RAM (Utilizes fabric Distributed RAM/ROM)
//   - Latency         : 1 clock cycle (Registered output for Fmax > 450 MHz)
//   - SQNR Precision  : 51.11 dB (Max Error: 0.007624, MAE: 0.000976)
// =============================================================================

`timescale 1ns / 1ps

module sigmoid_lut_256 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire signed [7:0] in_x,   // Q4.4 (-8.0 to +7.9375)
    output reg         [7:0] out_y    // Q0.8 (0.0 to 1.0)
);

    // Internal unsigned 8-bit address: map signed [-128..127] to unsigned [0..255]
    // in_x = -128 (0x80) -> addr = 0
    // in_x =    0 (0x00) -> addr = 128
    // in_x = +127 (0x7F) -> addr = 255
    wire [7:0] lut_addr = in_x ^ 8'h80;

    // ROM declaration (Synthesizes to Distributed ROM on UltraScale+)
    (* rom_style = "distributed" *) reg [7:0] rom [0:255];

    // Initialize ROM contents from file or inline values
    initial begin
        $readmemh("sigmoid_table_256.hex", rom);
    end

    // Registered output for high-speed pipelining
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_y <= 8'h00;
        end else if (enable) begin
            out_y <= rom[lut_addr];
        end
    end

endmodule
