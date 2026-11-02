// Fixed-point joint setpoint ramp. Values are signed application fixed-point.
// Target updates are applied on a control tick; step_size must be positive.
module linear_interpolator #(
    parameter integer CLOCK_HZ=100000000,
    parameter integer CONTROL_HZ=1000
)(
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire signed [31:0] target,
    input wire signed [31:0] step_size,
    output reg signed [31:0] position,
    output reg tick
);
    localparam integer DIVISOR=CLOCK_HZ/CONTROL_HZ;
    reg [31:0] divider;
    wire signed [32:0] delta={target[31],target}-{position[31],position};
    wire signed [32:0] max_step={1'b0,step_size};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin divider<=0; position<=0; tick<=0; end
        else begin
            tick<=0;
            if (divider==DIVISOR-1) begin
                divider<=0;
                tick<=1;
                if (enable && step_size>0) begin
                    if (delta>max_step) position<=position+step_size;
                    else if (delta< -max_step) position<=position-step_size;
                    else position<=target;
                end
            end else divider<=divider+1;
        end
    end
endmodule
