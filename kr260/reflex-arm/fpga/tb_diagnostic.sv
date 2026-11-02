`timescale 1ns/1ps
module tb_diagnostic;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] control=0;
    wire [31:0] status,lo,hi,ticks,period,hb,fc,pos;
    diagnostic_core dut(clk,rst_n,control,status,lo,hi,ticks,period,hb,fc,pos);
    initial begin
        #1;rst_n=0;@(posedge clk);#1;@(negedge clk);rst_n=1;
        repeat(210000) @(negedge clk);
        if(ticks<2 || period!=100000) $fatal(1,"hardware tick counters failed");
        control=1;repeat(3)@(negedge clk);control=2;
        repeat(3)@(negedge clk);
        if(status[2:0]!=3) $fatal(1,"FSM arm wiring failed");
        control=8;repeat(3)@(negedge clk);
        if(status[7:4]!=1 || status[2:1]!=2 || status[0]!=0 || fc==0) $fatal(1,"fault timestamp wiring failed");
        $display("PASS diagnostic signature counters heartbeat arm fault timestamp");
        $finish;
    end
endmodule
