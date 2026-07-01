`timescale 1ns / 1ps
`default_nettype none

//=========================================================
// 3x3 PARALLEL MAC ACCELERATOR - FIXED HANDSHAKE v2
// Uses 64-bit AXI-Stream (8 bytes per cycle)
//=========================================================

module mac_3x3_accelerator(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis" *)
    input  wire        s_axis_aclk,
    input  wire        s_axis_aresetn,

    // AXI-Stream Slave (64-bit data)
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tlast,

    // AXI-Stream Master (32-bit result)
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire [31:0] m_axis_tdata,
    output wire        m_axis_tlast,
    output wire [3:0]  m_axis_tkeep
);

    // ============================================================
    // State Machine
    // ============================================================
    localparam S_IDLE    = 2'b00,
               S_RECEIVE = 2'b01,
               S_OUTPUT  = 2'b10;

    reg [1:0]  state;
    reg [63:0] data_reg [0:2];
    reg [1:0]  beat_count;

    // FIX: Ready in BOTH IDLE and RECEIVE states
    assign s_axis_tready = (state == S_IDLE) || (state == S_RECEIVE);

    // ============================================================
    // Main Control State Machine - FIXED
    // ============================================================
    always @(posedge s_axis_aclk) begin
        if (!s_axis_aresetn) begin
            state      <= S_IDLE;
            beat_count <= 2'b00;
        end else begin
            case (state)
                S_IDLE: begin
                    // FIX: Capture first beat immediately when valid & ready
                    if (s_axis_tvalid && s_axis_tready) begin
                        data_reg[0] <= s_axis_tdata;
                        beat_count  <= 2'b01;  // Next will be beat 1
                        state       <= S_RECEIVE;
                    end
                end

                S_RECEIVE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        data_reg[beat_count] <= s_axis_tdata;
                        beat_count           <= beat_count + 1'b1;
                        
                        if (s_axis_tlast) begin
                            state      <= S_OUTPUT;
                            beat_count <= 2'b00;
                        end
                    end
                end

                S_OUTPUT: begin
                    if (m_axis_tvalid && m_axis_tready) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // [Rest of your Verilog unchanged: byte extraction, multipliers, adder tree, output assignments]
    // ... keep d0-d17, x0-x8, y0-y8, baugh instances, CLA_24 instances, output assignments exactly as before ...

    // ============================================================
    // Extract 9 x values and 9 y values from 3x64-bit registers
    // ============================================================
    wire [7:0] d0 = data_reg[0][7:0];
    wire [7:0] d1 = data_reg[0][15:8];
    wire [7:0] d2 = data_reg[0][23:16];
    wire [7:0] d3 = data_reg[0][31:24];
    wire [7:0] d4 = data_reg[0][39:32];
    wire [7:0] d5 = data_reg[0][47:40];
    wire [7:0] d6 = data_reg[0][55:48];
    wire [7:0] d7 = data_reg[0][63:56];

    wire [7:0] d8  = data_reg[1][7:0];
    wire [7:0] d9  = data_reg[1][15:8];
    wire [7:0] d10 = data_reg[1][23:16];
    wire [7:0] d11 = data_reg[1][31:24];
    wire [7:0] d12 = data_reg[1][39:32];
    wire [7:0] d13 = data_reg[1][47:40];
    wire [7:0] d14 = data_reg[1][55:48];
    wire [7:0] d15 = data_reg[1][63:56];

    wire [7:0] d16 = data_reg[2][7:0];
    wire [7:0] d17 = data_reg[2][15:8];

    // 9 x values
    wire signed [7:0] x0 = $signed(d0);
    wire signed [7:0] x1 = $signed(d1);
    wire signed [7:0] x2 = $signed(d2);
    wire signed [7:0] x3 = $signed(d3);
    wire signed [7:0] x4 = $signed(d4);
    wire signed [7:0] x5 = $signed(d5);
    wire signed [7:0] x6 = $signed(d6);
    wire signed [7:0] x7 = $signed(d7);
    wire signed [7:0] x8 = $signed(d8);

    // 9 y values
    wire signed [7:0] y0 = $signed(d9);
    wire signed [7:0] y1 = $signed(d10);
    wire signed [7:0] y2 = $signed(d11);
    wire signed [7:0] y3 = $signed(d12);
    wire signed [7:0] y4 = $signed(d13);
    wire signed [7:0] y5 = $signed(d14);
    wire signed [7:0] y6 = $signed(d15);
    wire signed [7:0] y7 = $signed(d16);
    wire signed [7:0] y8 = $signed(d17);

    // ============================================================
    // Combinational MAC - 9 Multipliers + Adder Tree
    // ============================================================
    wire signed [15:0] mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7, mul8;
    wire signed [23:0] sum_out;

    baugh ba0 (.a(x0), .b(y0), .p(mul0));
    baugh ba1 (.a(x1), .b(y1), .p(mul1));
    baugh ba2 (.a(x2), .b(y2), .p(mul2));
    baugh ba3 (.a(x3), .b(y3), .p(mul3));
    baugh ba4 (.a(x4), .b(y4), .p(mul4));
    baugh ba5 (.a(x5), .b(y5), .p(mul5));
    baugh ba6 (.a(x6), .b(y6), .p(mul6));
    baugh ba7 (.a(x7), .b(y7), .p(mul7));
    baugh ba8 (.a(x8), .b(y8), .p(mul8));

    wire signed [23:0] mul0_24 = {{8{mul0[15]}}, mul0};
    wire signed [23:0] mul1_24 = {{8{mul1[15]}}, mul1};
    wire signed [23:0] mul2_24 = {{8{mul2[15]}}, mul2};
    wire signed [23:0] mul3_24 = {{8{mul3[15]}}, mul3};
    wire signed [23:0] mul4_24 = {{8{mul4[15]}}, mul4};
    wire signed [23:0] mul5_24 = {{8{mul5[15]}}, mul5};
    wire signed [23:0] mul6_24 = {{8{mul6[15]}}, mul6};
    wire signed [23:0] mul7_24 = {{8{mul7[15]}}, mul7};
    wire signed [23:0] mul8_24 = {{8{mul8[15]}}, mul8};

    wire signed [23:0] s1, s2, s3, s4, s5, s6, s7;
    
    CLA_24 cla1 (.a(mul0_24), .b(mul1_24), .cin(1'b0), .s(s1), .cout());
    CLA_24 cla2 (.a(mul2_24), .b(mul3_24), .cin(1'b0), .s(s2), .cout());
    CLA_24 cla3 (.a(mul4_24), .b(mul5_24), .cin(1'b0), .s(s3), .cout());
    CLA_24 cla4 (.a(mul6_24), .b(mul7_24), .cin(1'b0), .s(s4), .cout());
    
    CLA_24 cla5 (.a(s1), .b(s2), .cin(1'b0), .s(s5), .cout());
    CLA_24 cla6 (.a(s3), .b(s4), .cin(1'b0), .s(s6), .cout());
    
    CLA_24 cla7 (.a(s5), .b(s6), .cin(1'b0), .s(s7), .cout());
    CLA_24 cla8 (.a(s7), .b(mul8_24), .cin(1'b0), .s(sum_out), .cout());

    // ============================================================
    // AXI-Stream Master Interface Output Assignments
    // ============================================================
    assign m_axis_tvalid = (state == S_OUTPUT);
    assign m_axis_tdata  = {{8{sum_out[23]}}, sum_out};
    assign m_axis_tlast  = m_axis_tvalid;
    assign m_axis_tkeep  = 4'b1111;

endmodule
//=========================================================
// BAUGH WOOLLEY MULTIPLIER - 8-bit signed
//=========================================================
module baugh(
    input  wire signed [7:0]  a, b,
    output wire signed [15:0] p
);
    // Partial products generation logic
    wire p00 = a[0] & b[0]; wire p10 = a[1] & b[0]; wire p20 = a[2] & b[0];
    wire p30 = a[3] & b[0]; wire p40 = a[4] & b[0]; wire p50 = a[5] & b[0];
    wire p60 = a[6] & b[0];

    wire p01 = a[0] & b[1]; wire p11 = a[1] & b[1]; wire p21 = a[2] & b[1];
    wire p31 = a[3] & b[1]; wire p41 = a[4] & b[1]; wire p51 = a[5] & b[1];
    wire p61 = a[6] & b[1];

    wire p02 = a[0] & b[2]; wire p12 = a[1] & b[2]; wire p22 = a[2] & b[2];
    wire p32 = a[3] & b[2]; wire p42 = a[4] & b[2]; wire p52 = a[5] & b[2];
    wire p62 = a[6] & b[2];

    wire p03 = a[0] & b[3]; wire p13 = a[1] & b[3]; wire p23 = a[2] & b[3];
    wire p33 = a[3] & b[3]; wire p43 = a[4] & b[3]; wire p53 = a[5] & b[3];
    wire p63 = a[6] & b[3];

    wire p04 = a[0] & b[4]; wire p14 = a[1] & b[4]; wire p24 = a[2] & b[4];
    wire p34 = a[3] & b[4]; wire p44 = a[4] & b[4]; wire p54 = a[5] & b[4];
    wire p64 = a[6] & b[4];

    wire p05 = a[0] & b[5]; wire p15 = a[1] & b[5]; wire p25 = a[2] & b[5];
    wire p35 = a[3] & b[5]; wire p45 = a[4] & b[5]; wire p55 = a[5] & b[5];
    wire p65 = a[6] & b[5];

    wire p06 = a[0] & b[6]; wire p16 = a[1] & b[6]; wire p26 = a[2] & b[6];
    wire p36 = a[3] & b[6]; wire p46 = a[4] & b[6]; wire p56 = a[5] & b[6];
    wire p66 = a[6] & b[6];

    wire p77 = a[7] & b[7];

    wire p70 = ~(a[7] & b[0]); wire p71 = ~(a[7] & b[1]);
    wire p72 = ~(a[7] & b[2]); wire p73 = ~(a[7] & b[3]);
    wire p74 = ~(a[7] & b[4]); wire p75 = ~(a[7] & b[5]);
    wire p76 = ~(a[7] & b[6]);

    wire p07 = ~(a[0] & b[7]); wire p17 = ~(a[1] & b[7]);
    wire p27 = ~(a[2] & b[7]); wire p37 = ~(a[3] & b[7]);
    wire p47 = ~(a[4] & b[7]); wire p57 = ~(a[5] & b[7]);
    wire p67 = ~(a[6] & b[7]);

    assign p[0] = p00;

    wire s1_1, c1_1;
    ha ha1_1(.a(p10), .b(p01), .s(s1_1), .c(c1_1));
    assign p[1] = s1_1;

    wire s1_2, c1_2, c2_2;
    fa fa1_2(.a(p20), .b(p11), .c(p02), .s(s1_2), .cout(c1_2));
    ha ha2_2(.a(s1_2), .b(c1_1), .s(p[2]), .c(c2_2));

    wire s1_3, c1_3, s2_3, c2_3, c3_3;
    fa fa1_3(.a(p30), .b(p21), .c(p12), .s(s1_3), .cout(c1_3));
    fa fa2_3(.a(s1_3), .b(p03), .c(c1_2), .s(s2_3), .cout(c2_3));
    ha ha3_3(.a(s2_3), .b(c2_2), .s(p[3]), .c(c3_3));

    wire s1_4, c1_4, s2_4, c2_4, s3_4, c3_4, c4_4;
    fa fa1_4(.a(p40), .b(p31), .c(p22), .s(s1_4), .cout(c1_4));
    fa fa2_4(.a(s1_4), .b(p13), .c(p04), .s(s2_4), .cout(c2_4));
    fa fa3_4(.a(s2_4), .b(c1_3), .c(c2_3), .s(s3_4), .cout(c3_4));
    ha ha4_4(.a(s3_4), .b(c3_3), .s(p[4]), .c(c4_4));

    wire s1_5, c1_5, s2_5, c2_5, s3_5, c3_5, s4_5, c4_5, c5_5;
    fa fa1_5(.a(p50), .b(p41), .c(p32), .s(s1_5), .cout(c1_5));
    fa fa2_5(.a(s1_5), .b(p23), .c(p14), .s(s2_5), .cout(c2_5));
    fa fa3_5(.a(s2_5), .b(p05), .c(c1_4), .s(s3_5), .cout(c3_5));
    fa fa4_5(.a(s3_5), .b(c2_4), .c(c3_4), .s(s4_5), .cout(c4_5));
    ha ha5_5(.a(s4_5), .b(c4_4), .s(p[5]), .c(c5_5));

    wire s1_6, c1_6, s2_6, c2_6, s3_6, c3_6, s4_6, c4_6, s5_6, c5_6, c6_6;
    fa fa1_6(.a(p60), .b(p51), .c(p42), .s(s1_6), .cout(c1_6));
    fa fa2_6(.a(s1_6), .b(p33), .c(p24), .s(s2_6), .cout(c2_6));
    fa fa3_6(.a(s2_6), .b(p15), .c(p06), .s(s3_6), .cout(c3_6));
    fa fa4_6(.a(s3_6), .b(c1_5), .c(c2_5), .s(s4_6), .cout(c4_6));
    fa fa5_6(.a(s4_6), .b(c3_5), .c(c4_5), .s(s5_6), .cout(c5_6));
    ha ha6_6(.a(s5_6), .b(c5_5), .s(p[6]), .c(c6_6));

    wire s1_7, c1_7, s2_7, c2_7, s3_7, c3_7, s4_7, c4_7;
    wire s5_7, c5_7, s6_7, c6_7, s7_7, c7_7, s7_7_dup, c7_7_dup;
    fa fa1_7(.a(p70), .b(p61), .c(p52), .s(s1_7), .cout(c1_7));
    fa fa2_7(.a(s1_7), .b(p43), .c(p34), .s(s2_7), .cout(c2_7));
    fa fa3_7(.a(s2_7), .b(p25), .c(p16), .s(s3_7), .cout(c3_7));
    fa fa4_7(.a(s3_7), .b(p07), .c(c1_6), .s(s4_7), .cout(c4_7));
    fa fa5_7(.a(s4_7), .b(c2_6), .c(c3_6), .s(s5_7), .cout(c5_7));
    fa fa6_7(.a(s5_7), .b(c4_6), .c(c5_6), .s(s6_7), .cout(c6_7));
    fa fa7_7(.a(s6_7), .b(c6_6), .c(1'b1), .s(s7_7), .cout(c7_7));
    ha ha7_7_inst(.a(s7_7), .b(1'b1), .s(p[7]), .c(c7_7_dup));

    wire s1_8, c1_8, s2_8, c2_8, s3_8, c3_8, s4_8, c4_8;
    wire s5_8, c5_8, s6_8, c6_8, c7_8;
    fa fa1_8 (.a(p71), .b(p62), .c(p53), .s(s1_8),  .cout(c1_8));
    fa fa2_8 (.a(s1_8),  .b(p44), .c(p35), .s(s2_8),  .cout(c2_8));
    fa fa3_8 (.a(s2_8),  .b(p26), .c(p17), .s(s3_8),  .cout(c3_8));
    fa fa4_8 (.a(s3_8),  .b(c1_7), .c(c2_7), .s(s4_8),  .cout(c4_8));
    fa fa5_8 (.a(s4_8),  .b(c3_7), .c(c4_7), .s(s5_8),  .cout(c5_8));
    fa fa6_8 (.a(s5_8),  .b(c5_7), .c(c6_7), .s(s6_8),  .cout(c6_8));
    fa fa7_8 (.a(s6_8),  .b(c7_7), .c(c7_7_dup), .s(p[8]), .cout(c7_8));

    wire s1_9, c1_9, s2_9, c2_9, s3_9, c3_9, s4_9, c4_9, s5_9, c5_9, c6_9;
    fa fa1_9 (.a(p72), .b(p63), .c(p54), .s(s1_9),  .cout(c1_9));
    fa fa2_9 (.a(s1_9),  .b(p45), .c(p36), .s(s2_9),  .cout(c2_9));
    fa fa3_9 (.a(s2_9),  .b(p27), .c(c1_8), .s(s3_9),  .cout(c3_9));
    fa fa4_9 (.a(s3_9),  .b(c2_8), .c(c3_8), .s(s4_9),  .cout(c4_9));
    fa fa5_9 (.a(s4_9),  .b(c4_8), .c(c5_8), .s(s5_9),  .cout(c5_9));
    fa fa6_9 (.a(s5_9),  .b(c6_8), .c(c7_8), .s(p[9]),  .cout(c6_9));

    wire s1_10, c1_10, s2_10, c2_10, s3_10, c3_10, s4_10, c4_10, c5_10;
    fa fa1_10(.a(p73), .b(p64), .c(p55), .s(s1_10), .cout(c1_10));
    fa fa2_10(.a(s1_10), .b(p46), .c(p37), .s(s2_10), .cout(c2_10));
    fa fa3_10(.a(s2_10), .b(c1_9), .c(c2_9), .s(s3_10), .cout(c3_10));
    fa fa4_10(.a(s3_10), .b(c3_9), .c(c4_9), .s(s4_10), .cout(c4_10));
    fa fa5_10(.a(s4_10), .b(c5_9), .c(c6_9), .s(p[10]), .cout(c5_10));

    wire s1_11, c1_11, s2_11, c2_11, s3_11, c3_11, c4_11;
    fa fa1_11(.a(p74), .b(p65), .c(p56), .s(s1_11), .cout(c1_11));
    fa fa2_11(.a(s1_11), .b(p47), .c(c1_10), .s(s2_11), .cout(c2_11));
    fa fa3_11(.a(s2_11), .b(c2_10), .c(c3_10), .s(s3_11), .cout(c3_11));
    fa fa4_11(.a(s3_11), .b(c4_10), .c(c5_10), .s(p[11]), .cout(c4_11));

    wire s1_12, c1_12, s2_12, c2_12, c3_12;
    fa fa1_12(.a(p75), .b(p66), .c(p57), .s(s1_12), .cout(c1_12));
    fa fa2_12(.a(s1_12), .b(c1_11), .c(c2_11), .s(s2_12), .cout(c2_12));
    fa fa3_12(.a(s2_12), .b(c3_11), .c(c4_11), .s(p[12]), .cout(c3_12));

    wire s1_13, c1_13, c2_13;
    fa fa1_13(.a(p76), .b(p67), .c(c1_12), .s(s1_13), .cout(c1_13));
    fa fa2_13(.a(s1_13), .b(c2_12), .c(c3_12), .s(p[13]), .cout(c2_13));

    wire s1_14, c1_14, s2_14, c2_14, c3_14;
    fa fa1_14(.a(p77), .b(c1_13), .c(c2_13), .s(s1_14), .cout(c1_14));
    ha ha1_14(.a(s1_14), .b(1'b1), .s(s2_14), .c(c2_14));
    ha ha2_14(.a(s2_14), .b(1'b1), .s(p[14]), .c(c3_14));

    wire cout_final;
    fa fa1_15(.a(c1_14), .b(c2_14), .c(c3_14), .s(p[15]), .cout(cout_final));

endmodule

//=========================================================
// CLA_24 - Carry Lookahead Adder (24-bit)
//=========================================================
module CLA_24 (
    input  wire signed [23:0] a,
    input  wire signed [23:0] b,
    input  wire        cin,
    output wire signed [23:0] s,
    output wire        cout
);
    wire [23:0] g = a & b;
    wire [23:0] p = a ^ b;
    wire [24:0] c;

    assign c[0] = cin;
    assign c[1] = g[0]  | (p[0]  & c[0]);
    assign c[2] = g[1]  | (p[1]  & g[0])  | (p[1]  & p[0]  & c[0]);
    assign c[3] = g[2]  | (p[2]  & g[1])  | (p[2]  & p[1]  & g[0])  | (p[2]  & p[1]  & p[0]  & c[0]);
    assign c[4] = g[3]  | (p[3]  & g[2])  | (p[3]  & p[2]  & g[1])  | (p[3]  & p[2]  & p[1]  & g[0])  | (p[3]  & p[2]  & p[1]  & p[0]  & c[0]);

    assign c[5]  = g[4]  | (p[4]  & c[4]);
    assign c[6]  = g[5]  | (p[5]  & g[4])  | (p[5]  & p[4]  & c[4]);
    assign c[7]  = g[6]  | (p[6]  & g[5])  | (p[6]  & p[5]  & g[4])  | (p[6]  & p[5]  & p[4]  & c[4]);
    assign c[8]  = g[7]  | (p[7]  & g[6])  | (p[7]  & p[6]  & g[5])  | (p[7]  & p[6]  & p[5]  & g[4])  | (p[7]  & p[6]  & p[5]  & p[4]  & c[4]);

    assign c[9]  = g[8]  | (p[8]  & c[8]);
    assign c[10] = g[9]  | (p[9]  & g[8])  | (p[9]  & p[8]  & c[8]);
    assign c[11] = g[10] | (p[10] & g[9])  | (p[10] & p[9]  & g[8])  | (p[10] & p[9]  & p[8]  & c[8]);
    assign c[12] = g[11] | (p[11] & g[10]) | (p[11] & p[10] & g[9])  | (p[11] & p[10] & p[9]  & g[8])  | (p[11] & p[10] & p[9]  & p[8]  & c[8]);

    assign c[13] = g[12] | (p[12] & c[12]);
    assign c[14] = g[13] | (p[13] & g[12]) | (p[13] & p[12] & c[12]);
    assign c[15] = g[14] | (p[14] & g[13]) | (p[14] & p[13] & g[12]) | (p[14] & p[13] & p[12] & c[12]);
    assign c[16] = g[15] | (p[15] & g[14]) | (p[15] & p[14] & g[13]) | (p[15] & p[14] & p[13] & g[12]) | (p[15] & p[14] & p[13] & p[12] & c[12]);

    assign c[17] = g[16] | (p[16] & c[16]);
    assign c[18] = g[17] | (p[17] & g[16]) | (p[17] & p[16] & c[16]);
    assign c[19] = g[18] | (p[18] & g[17]) | (p[18] & p[17] & g[16]) | (p[18] & p[17] & p[16] & c[16]);
    assign c[20] = g[19] | (p[19] & g[18]) | (p[19] & p[18] & g[17]) | (p[19] & p[18] & p[17] & g[16]) | (p[19] & p[18] & p[17] & p[16] & c[16]);

    assign c[21] = g[20] | (p[20] & c[20]);
    assign c[22] = g[21] | (p[21] & g[20]) | (p[21] & p[20] & c[20]);
    assign c[23] = g[22] | (p[22] & g[21]) | (p[22] & p[21] & g[20]) | (p[22] & p[21] & p[20] & c[20]);
    assign c[24] = g[23] | (p[23] & g[22]) | (p[23] & p[22] & g[21]) | (p[23] & p[22] & p[21] & g[20]) | (p[23] & p[22] & p[21] & p[20] & c[20]);

    assign s = p ^ c[23:0];
    assign cout = c[24];
endmodule

//=========================================================
// HA - Half Adder
//=========================================================
module ha(
    input  wire a, b,
    output wire s, c
);
    assign s = a ^ b;
    assign c = a & b;
endmodule

//=========================================================
// FA - Full Adder
//=========================================================
module fa(
    input  wire a, b, c,
    output wire s, cout
);
    assign s = a ^ b ^ c;
    assign cout = (a & b) | (b & c) | (c & a);
endmodule

`default_nettype wire
