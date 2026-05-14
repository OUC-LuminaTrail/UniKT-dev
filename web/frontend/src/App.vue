<template>
  <el-config-provider :locale="zhCn">
    <AppLayout>
      <router-view />
    </AppLayout>
  </el-config-provider>
</template>

<script setup lang="ts">
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import AppLayout from '@/components/layout/AppLayout.vue'
</script>
