module tb_vision_axis;
 reg clk=0;always #5 clk=~clk;
 reg resetn=0;reg[31:0] din=0;reg valid=0,last=0;reg[3:0] keep=15;
 wire ready;wire[31:0] dout;wire ov,ol;wire[3:0] ok;reg rd=0;wire[31:0] sig,frames;
 vision_axis d(clk,resetn,din,valid,ready,last,keep,dout,ov,rd,ol,ok,sig,frames);
 reg [7:0] expected[0:20000];reg expected_last[0:20000];integer sent=0,received=0,j,f,n,c,seed=19;
 reg[31:0] word;integer r,g,b,l;reg stalled=0;reg[36:0] held;
 always @(negedge clk) if(resetn)rd=($random(seed)&3)!=0;
 always @(posedge clk) if(resetn) begin
   if(stalled && {dout,ol,ok}!==held)$fatal(1,"unstable backpressure");
   stalled=ov&&!rd;held={dout,ol,ok};
   if(ov&&rd)begin
    for(c=0;c<4;c=c+1)if(ok[c])begin
     if(dout[c*8+:8]!==expected[received])$fatal(1,"label mismatch %d",received);
     if(expected_last[received] !== (ol && (c==3 || !ok[c+1])))$fatal(1,"TLAST mismatch");
     received=received+1;
    end
   end
 end
 initial begin
  repeat(4)@(negedge clk);resetn=1;
  for(f=1;f<=25;f=f+1)begin
   n=f<10?f:100+f*7;
   for(j=0;j<n;j=j+1)begin
    word=$random(seed);r=word[7:0];g=word[15:8];b=word[23:16];l=0;
    if(word[31:24])begin
     if(r>50 && r>2*g && r>2*b)l=1;
     else if(b>50 && b>2*r && b>2*g)l=2;
     else if(r>50&&g>40&&2*b<r&&2*b<g&&r<2*g&&g<2*r)l=3;
     else if(r>50&&b>40&&2*g<r&&2*g<b&&r<2*b&&b<2*r)l=4;
    end
    expected[sent]=l;expected_last[sent]=(j==n-1);sent=sent+1;
    @(negedge clk);din=word;valid=1;last=(j==n-1);
    @(posedge clk);while(!ready)@(posedge clk);
    @(negedge clk);valid=0;last=0;
    repeat($unsigned($random(seed))%3)@(negedge clk);
   end
  end
  repeat(30)@(negedge clk);
  if(received!=sent||frames!=25||sig!=32'h52444d31)$fatal(1,"count/identity mismatch");
  $display("PASS AXIS %d pixels 25 frames randomized backpressure and partial final words",received);$finish;
 end
 initial begin #2000000;$fatal(1,"timeout");end
endmodule
