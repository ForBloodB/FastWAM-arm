#include <stdint.h>
#include <stddef.h>
// Native MMIO transport only. Addresses are validated by the Python UIO owner.
static int send(volatile uint32_t *in, volatile uint32_t *out, uint32_t data,
                uint32_t *token, uint32_t *status) {
  *token ^= 0x80000000u; in[0]=data|*token; __sync_synchronize();
  for(int k=0;k<1000000;k++) {
    uint32_t s=out[0];
    if((s&0x80000000u)==*token){*status=s;return 0;}
  }
  return -1;
}
int vision_frame(volatile uint32_t *in, volatile uint32_t *out,
                 const uint8_t *rgba, uint8_t *labels, int w,int h,uint32_t *stats) {
  if(w<1||w>640||h<1||h>480)return -2;
  uint32_t token=out[0]&0x80000000u,status=0;
  if(send(in,out,0x40000000u|((uint32_t)h<<12)|w,&token,&status))return -3;
  if(status&0x20000000u)return -4;
  for(int i=0;i<w*h;i++) {
    uint32_t pixel=((uint32_t)rgba[4*i]<<16)|((uint32_t)rgba[4*i+1]<<8)|rgba[4*i+2];
    if(rgba[4*i+3])pixel|=0x01000000u;
    if(send(in,out,pixel,&token,&status))return -5;
    if(status&0x20000000u)return -6;
    if(((status>>3)&0xfffffu)!=(uint32_t)i+1)return -7;
    labels[i]=status&7;
  }
  if(send(in,out,0x20000000u,&token,&status))return -8;
  if((status&0x60000000u)!=0x40000000u)return -9;
  for(uint32_t i=0;i<28;i++) {
    in[2]=i;__sync_synchronize();(void)out[2];stats[i]=out[2];
  }
  return 0;
}
