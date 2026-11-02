#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x2afb4b93, "module_layout" },
	{ 0x62b5eed9, "platform_driver_unregister" },
	{ 0x21936014, "__platform_driver_register" },
	{ 0x8da6585d, "__stack_chk_fail" },
	{ 0xb788fb30, "gic_pmr_sync" },
	{ 0x908e5601, "cpu_hwcaps" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x4a3ad70e, "wait_for_completion_timeout" },
	{ 0xf888ca21, "sg_init_table" },
	{ 0x89940875, "mutex_lock_interruptible" },
	{ 0x12a4e128, "__arch_copy_from_user" },
	{ 0x800473f, "__cond_resched" },
	{ 0xc7237a3e, "misc_deregister" },
	{ 0x69f38847, "cpu_hwcap_keys" },
	{ 0x14b89635, "arm64_const_caps_ready" },
	{ 0xd4a7ae0d, "_dev_info" },
	{ 0x6bc6ef31, "misc_register" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x608741b5, "__init_swait_queue_head" },
	{ 0x248377fd, "dma_release_channel" },
	{ 0xa3dbade0, "dma_free_attrs" },
	{ 0xfa41359d, "dma_alloc_attrs" },
	{ 0x55d32237, "dma_request_chan" },
	{ 0xdf25a54f, "devm_kmalloc" },
	{ 0xa6257a2f, "complete" },
	{ 0xcfa94a2e, "dma_mmap_attrs" },
	{ 0x4b0a3f52, "gic_nonsecure_priorities" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("of:N*T*Creflex,image-dma");
MODULE_ALIAS("of:N*T*Creflex,image-dmaC*");

MODULE_INFO(srcversion, "1063E7762A8E09B6C423D23");
