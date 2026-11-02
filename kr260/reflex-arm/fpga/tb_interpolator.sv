`timescale 1ns/1ps
module tb_interpolator;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,enable=1;
    reg signed [31:0] target=25,step_size=10;
    wire signed [31:0] pos;
    wire tick;
    linear_interpolator #(.CLOCK_HZ(100),.CONTROL_HZ(10)) dut(clk,rst_n,enable,target,step_size,pos,tick);
    integer cycles=0, previous=0, nticks=0;
    initial begin
        #1; rst_n=0; @(posedge clk); #1; @(negedge clk); rst_n=1;
        repeat(30) begin
            @(posedge clk); #1; cycles=cycles+1;
            if(tick) begin
                if(cycles-previous!=10) $fatal(1,"nonperiodic tick");
                previous=cycles; nticks=nticks+1;
            end
        end
        if(pos!==25 || nticks!=3) $fatal(1,"ramp or target clamp failed");
        enable=0; target=-25;
        repeat(20) begin @(posedge clk); #1; end
        if(pos!==25) $fatal(1,"disable did not hold");
        enable=1;
        repeat(50) begin @(posedge clk); #1; end
        if(pos!==-25) $fatal(1,"negative ramp failed");
        $display("PASS interpolation fixed period positive negative clamp hold");
        $finish;
    end
endmodule
