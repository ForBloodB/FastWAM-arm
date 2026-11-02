#include <stdint.h>
#include <stddef.h>
// Same byte layout and predicates as vision_axis.sv; scalar CPU baseline.
void image_labels(const uint8_t *rgba,uint8_t *out,size_t n){
 for(size_t i=0;i<n;i++){
  unsigned r=rgba[4*i],g=rgba[4*i+1],b=rgba[4*i+2];unsigned char x=0;
  if(rgba[4*i+3]){
   if(r>50&&r>2*g&&r>2*b)x=1;
   else if(b>50&&b>2*r&&b>2*g)x=2;
   else if(r>50&&g>40&&2*b<r&&2*b<g&&r<2*g&&g<2*r)x=3;
   else if(r>50&&b>40&&2*g<r&&2*g<b&&r<2*b&&b<2*r)x=4;
  }out[i]=x;
 }
}
