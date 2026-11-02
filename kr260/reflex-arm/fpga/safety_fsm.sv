// Prototype synchronous core. Board wrapper MUST synchronize/debounce inputs,
// constrain timing and wire a separately verified physical brake/STO circuit.
// heartbeat is a one-clock accepted-command pulse, not an unvalidated packet.
module safety_fsm #(
    parameter integer WATCHDOG_CYCLES = 10000000
)(
    input wire clk,
    input wire rst_n,
    input wire heartbeat,
    input wire arm_request,
    input wire manual_reset,
    input wire estop,
    input wire limit_hit,
    input wire sensor_ok,
    input wire command_ok,
    output wire drive_enable,
    output reg [1:0] state,
    output reg [3:0] fault_code
);
    localparam DISARMED=2'd0, ARMED=2'd1, FAULT=2'd2;
    reg [31:0] age;
    reg seen_heartbeat;
    wire fresh = heartbeat || (seen_heartbeat && age < WATCHDOG_CYCLES-1);
    wire bad = estop || limit_hit || !sensor_ok || !command_ok;
    // Immediate combinational inhibit in addition to the latched state.
    // A production e-stop additionally requires an independent physical path.
    assign drive_enable = rst_n && state==ARMED && !bad && fresh;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            age<=WATCHDOG_CYCLES;
            seen_heartbeat<=0;
            state<=DISARMED;
            fault_code<=0;
        end else begin
            if (heartbeat) begin age<=0; seen_heartbeat<=1; end
            else if (age < WATCHDOG_CYCLES) age<=age+1;
            if (bad || (state==ARMED && !fresh)) begin
                state<=FAULT;
                if (estop) fault_code<=1;
                else if (limit_hit) fault_code<=2;
                else if (!sensor_ok) fault_code<=3;
                else if (!command_ok) fault_code<=4;
                else fault_code<=5;
            end else case (state)
                FAULT: if (manual_reset && fresh && !arm_request) begin
                    state<=DISARMED; fault_code<=0;
                end
                DISARMED: if (arm_request && fresh) state<=ARMED;
                ARMED: state<=ARMED;
                default: begin state<=FAULT; fault_code<=15; end
            endcase
        end
    end
endmodule
