// RGBA/foreground -> packed 8-bit labels. One pixel per accepted input beat.
// TLAST/TKEEP propagate frame boundaries, including 1/2/3-byte final words.
module vision_axis(
 (* X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn, FREQ_HZ 99999001" *) input wire aclk,
 (* X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *) input wire aresetn,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TDATA" *) input wire [31:0] s_axis_tdata,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input wire s_axis_tvalid,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire s_axis_tready,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TLAST" *) input wire s_axis_tlast,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *) input wire [3:0] s_axis_tkeep,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output reg [31:0] m_axis_tdata,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output reg m_axis_tvalid,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input wire m_axis_tready,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output reg m_axis_tlast,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output reg [3:0] m_axis_tkeep,
 output wire [31:0] signature,output reg [31:0] frames);
 assign signature=32'h52444d31;
 wire [8:0] r={1'b0,s_axis_tdata[7:0]},g={1'b0,s_axis_tdata[15:8]},b={1'b0,s_axis_tdata[23:16]};
 reg [7:0] label;reg [1:0] pos;reg [31:0] packed_word;reg [31:0] assembled;
 always @* begin
  label=0;
  if(s_axis_tdata[31:24]!=0 && s_axis_tkeep==4'hf) begin
   if(r>50 && r>2*g && r>2*b) label=1;
   else if(b>50 && b>2*r && b>2*g) label=2;
   else if(r>50 && g>40 && 2*b<r && 2*b<g && r<2*g && g<2*r) label=3;
   else if(r>50 && b>40 && 2*g<r && 2*g<b && r<2*b && b<2*r) label=4;
  end
  assembled=packed_word;assembled[pos*8+:8]=label;
 end
 assign s_axis_tready=!m_axis_tvalid || m_axis_tready;
 always @(posedge aclk) begin
  if(!aresetn) begin
   m_axis_tvalid<=0;m_axis_tlast<=0;m_axis_tkeep<=0;m_axis_tdata<=0;
   packed_word<=0;pos<=0;frames<=0;
  end else begin
   if(m_axis_tvalid && m_axis_tready)m_axis_tvalid<=0;
   if(s_axis_tvalid && s_axis_tready) begin
    if(pos==3 || s_axis_tlast) begin
     m_axis_tdata<=assembled;m_axis_tvalid<=1;m_axis_tlast<=s_axis_tlast;
     m_axis_tkeep<=(4'b0001<<pos)|(4'b0001<<pos)-1;
     pos<=0;packed_word<=0;
     if(s_axis_tlast)frames<=frames+1;
    end else begin pos<=pos+1;packed_word<=assembled;end
   end
  end
 end
endmodule
