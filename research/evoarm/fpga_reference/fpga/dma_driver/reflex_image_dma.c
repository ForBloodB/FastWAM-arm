// SPDX-License-Identifier: GPL-2.0
// Image-only DMAengine client. Kernel owns DMA addresses and coherent buffers.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>
#include <linux/miscdevice.h>
#include <linux/completion.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/mm.h>
#define CAP (2*1024*1024)
#define RUN _IOW('R',1,__u32)
struct image_dma {struct dma_chan *tx,*rx;void *in,*out;dma_addr_t ia,oa;
 struct completion tc,rc;struct miscdevice misc;struct mutex lock;atomic_t opened;};
static void done(void *arg){complete(arg);}
static int opening(struct inode *ino,struct file *f){
 struct image_dma *d=container_of(f->private_data,struct image_dma,misc);
 if(atomic_cmpxchg(&d->opened,0,1))return -EBUSY;
 f->private_data=d;return 0;
}
static int closing(struct inode *ino,struct file *f){struct image_dma *d=f->private_data;atomic_set(&d->opened,0);return 0;}
static int mapping(struct file *f,struct vm_area_struct *v){
 struct image_dma *d=f->private_data;unsigned long offset=v->vm_pgoff;int rc;
 if(v->vm_end-v->vm_start>CAP || offset>1)return -EINVAL;
 v->vm_pgoff=0;
 rc=dma_mmap_coherent(offset?d->rx->device->dev:d->tx->device->dev,v,offset?d->out:d->in,offset?d->oa:d->ia,CAP);
 return rc;
}
static long command(struct file *f,unsigned int cmd,unsigned long arg){
 struct image_dma *d=f->private_data;struct dma_async_tx_descriptor *tx,*rx;dma_cookie_t ct,cr;
 __u32 n;int error=0;unsigned long timeout=msecs_to_jiffies(2000);
 if(cmd!=RUN)return -ENOTTY;
 if(copy_from_user(&n,(void __user *)arg,sizeof(n)))return -EFAULT;
 if(n==0 || n>640*480*4 || n%4)return -EINVAL;
 if(mutex_lock_interruptible(&d->lock))return -ERESTARTSYS;
 reinit_completion(&d->tc);reinit_completion(&d->rc);
 rx=dmaengine_prep_slave_single(d->rx,d->oa,n/4,DMA_DEV_TO_MEM,DMA_PREP_INTERRUPT|DMA_CTRL_ACK);
 tx=dmaengine_prep_slave_single(d->tx,d->ia,n,DMA_MEM_TO_DEV,DMA_PREP_INTERRUPT|DMA_CTRL_ACK);
 if(!tx||!rx){error=-EIO;goto stop;}
 rx->callback=done;rx->callback_param=&d->rc;tx->callback=done;tx->callback_param=&d->tc;
 cr=dmaengine_submit(rx);ct=dmaengine_submit(tx);
 if(dma_submit_error(cr)||dma_submit_error(ct)){error=-EIO;goto stop;}
 dma_wmb();dma_async_issue_pending(d->rx);dma_async_issue_pending(d->tx);
 if(!wait_for_completion_timeout(&d->rc,timeout)||!wait_for_completion_timeout(&d->tc,timeout)){error=-ETIMEDOUT;goto stop;}
 if(dma_async_is_tx_complete(d->rx,cr,NULL,NULL)!=DMA_COMPLETE || dma_async_is_tx_complete(d->tx,ct,NULL,NULL)!=DMA_COMPLETE){error=-EIO;goto stop;}
 dma_rmb();goto end;
stop:
 dmaengine_terminate_sync(d->tx);dmaengine_terminate_sync(d->rx);
end:
 mutex_unlock(&d->lock);return error;
}
static const struct file_operations ops={.owner=THIS_MODULE,.open=opening,.release=closing,.mmap=mapping,.unlocked_ioctl=command};
static int probe(struct platform_device *p){
 struct image_dma *d;int e;
 d=devm_kzalloc(&p->dev,sizeof(*d),GFP_KERNEL);if(!d)return -ENOMEM;
 d->tx=dma_request_chan(&p->dev,"tx");if(IS_ERR(d->tx))return PTR_ERR(d->tx);
 d->rx=dma_request_chan(&p->dev,"rx");if(IS_ERR(d->rx)){e=PTR_ERR(d->rx);goto free_tx;}
 d->in=dma_alloc_coherent(d->tx->device->dev,CAP,&d->ia,GFP_KERNEL);
 d->out=dma_alloc_coherent(d->rx->device->dev,CAP,&d->oa,GFP_KERNEL);
 if(!d->in||!d->out){e=-ENOMEM;goto free_mem;}
 init_completion(&d->tc);init_completion(&d->rc);mutex_init(&d->lock);atomic_set(&d->opened,0);
 d->misc.minor=MISC_DYNAMIC_MINOR;d->misc.name="reflex-image-dma";d->misc.fops=&ops;d->misc.mode=0600;
 e=misc_register(&d->misc);if(e)goto free_mem;
 platform_set_drvdata(p,d);dev_info(&p->dev,"bounded image DMA ready\n");return 0;
free_mem:
 if(d->in)dma_free_coherent(d->tx->device->dev,CAP,d->in,d->ia);
 if(d->out)dma_free_coherent(d->rx->device->dev,CAP,d->out,d->oa);
 dma_release_channel(d->rx);
free_tx:
 dma_release_channel(d->tx);return e;
}
static int removing(struct platform_device *p){
 struct image_dma *d=platform_get_drvdata(p);misc_deregister(&d->misc);
 dmaengine_terminate_sync(d->tx);dmaengine_terminate_sync(d->rx);
 dma_free_coherent(d->tx->device->dev,CAP,d->in,d->ia);dma_free_coherent(d->rx->device->dev,CAP,d->out,d->oa);
 dma_release_channel(d->tx);dma_release_channel(d->rx);return 0;
}
static const struct of_device_id match[]={ {.compatible="reflex,image-dma"},{}};
MODULE_DEVICE_TABLE(of,match);
static struct platform_driver reflex_image_driver={.probe=probe,.remove=removing,.driver={.name="reflex-image-dma",.of_match_table=match}};
module_platform_driver(reflex_image_driver);MODULE_LICENSE("GPL");MODULE_DESCRIPTION("Bounded image-only AXI DMA client");
