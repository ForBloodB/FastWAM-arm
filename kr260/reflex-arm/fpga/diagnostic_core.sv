// On-board diagnostic logic. drive_enable is internal and NEVER routed to CAN
// or any external actuator pin. Software fault inputs are for PL tests only.
module diagnostic_core(
    input wire clk,
    input wire rst_n,
    input wire [31:0] control,
    output wire [31:0] status,
    output wire [31:0] cycle_lo,
    output wire [31:0] cycle_hi,
    output reg [31:0] tick_count,
    output reg [31:0] last_period,
    output reg [31:0] heartbeat_cycle,
    output reg [31:0] fault_cycle,
    output wire signed [31:0] ramp_position
);
    reg [63:0] cycle_count;
    reg [31:0] previous_tick;
    reg previous_heartbeat;
    reg [1:0] previous_state;
    wire heartbeat=control[0] && !previous_heartbeat;
    wire [1:0] state;
    wire [3:0] fault_code;
    wire enabled,tick;
    assign cycle_lo=cycle_count[31:0];
    assign cycle_hi=cycle_count[63:32];
    assign status={16'h5246,8'd0,fault_code,1'b0,state,enabled};
    safety_fsm safety(clk,rst_n,heartbeat,control[1],control[2],
                      control[3],control[4],!control[5],!control[6],enabled,state,fault_code);
    linear_interpolator ramp(clk,rst_n,enabled,32'sd10000,32'sd1,ramp_position,tick);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count<=0;tick_count<=0;last_period<=0;previous_tick<=0;
            heartbeat_cycle<=0;fault_cycle<=0;previous_heartbeat<=0;previous_state<=0;
        end else begin
            cycle_count<=cycle_count+1;
            previous_heartbeat<=control[0];previous_state<=state;
            if (heartbeat) heartbeat_cycle<=cycle_count[31:0];
            if (state==2 && previous_state!=2) fault_cycle<=cycle_count[31:0];
            if (tick) begin
                tick_count<=tick_count+1;
                last_period<=cycle_count[31:0]-previous_tick;
                previous_tick<=cycle_count[31:0];
            end
        end
    end
endmodule
