<template>
  <el-menu :default-active="activeRoute" router class="sidebar-menu">
    <el-menu-item index="/tasks">
      <el-icon><Monitor /></el-icon>
      <span>训练任务</span>
    </el-menu-item>
    <el-menu-item index="/tasks/new">
      <el-icon><Plus /></el-icon>
      <span>新建任务</span>
    </el-menu-item>
    <el-menu-item index="/experiments">
      <el-icon><Folder /></el-icon>
      <span>实验记录</span>
    </el-menu-item>
    <el-menu-item index="/gpu">
      <el-icon><Monitor /></el-icon>
      <span>GPU 监控</span>
    </el-menu-item>
  </el-menu>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Monitor, Plus, Folder } from '@element-plus/icons-vue'

const route = useRoute()
const activeRoute = computed(() => route.path)
</script>

<style scoped>
.sidebar-menu {
  border-right: none;
  flex: 1;
}
</style>
