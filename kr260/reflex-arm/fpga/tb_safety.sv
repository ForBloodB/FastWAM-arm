`timescale 1ns/1ps
module tb_safety;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0,hb=0,arm=0,clear=0,estop=0,limit_hit=0,sensor_ok=1,cmd_ok=1;
    wire enable;
    wire [1:0] state;
    wire [3:0] code;
    safety_fsm #(.WATCHDOG_CYCLES(8)) dut(clk,rst_n,hb,arm,clear,estop,limit_hit,sensor_ok,cmd_ok,enable,state,code);
    task cycle; begin @(posedge clk); #1; end endtask
    task recover; begin
        @(negedge clk); arm=0; hb=1; clear=1; estop=0; limit_hit=0; sensor_ok=1; cmd_ok=1;
        cycle;
        if(state!==0 || enable!==0) $fatal(1,"recovery must DISARM");
        @(negedge clk); clear=0; arm=1;
        cycle;
        if(state!==1 || enable!==1) $fatal(1,"arm failed");
        @(negedge clk); hb=0; arm=0;
    end endtask
    initial begin
        $dumpfile("safety.vcd"); $dumpvars(0,tb_safety);
        #1; rst_n=0; cycle; @(negedge clk); rst_n=1;
        cycle; if(enable!==0) $fatal(1,"boot must disarm");
        arm=1; cycle; if(enable!==0) $fatal(1,"stale arm accepted");
        arm=0;
        recover;
        repeat(8) cycle;
        if(state!==2 || code!==5 || enable!==0) $fatal(1,"watchdog failed");
        recover;
        estop=1; #1; if(enable!==0) $fatal(1,"estop inhibit failed"); cycle;
        if(state!==2 || code!==1) $fatal(1,"estop latch failed");
        estop=0; hb=1; cycle; if(enable!==0) $fatal(1,"fault auto recovered");
        recover;
        limit_hit=1; cycle; if(code!==2 || enable!==0) $fatal(1,"limit failed");
        recover;
        sensor_ok=0; cycle; if(code!==3 || enable!==0) $fatal(1,"sensor failed");
        recover;
        cmd_ok=0; cycle; if(code!==4 || enable!==0) $fatal(1,"command failed");
        $display("PASS safety boot watchdog estop limit sensor command recovery");
        $finish;
    end
endmodule
