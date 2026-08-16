<template>
  <div class="selection-step-skeleton">
    <div class="sk-section">
      <el-skeleton-item variant="text" style="width:80px;margin-bottom:10px" />
      <el-skeleton-item variant="text" style="width:240px;height:32px" />
    </div>
    <div class="sk-section">
      <el-skeleton-item variant="text" style="width:48px;margin-bottom:10px" />
      <div class="sk-card-grid">
        <div v-for="i in 6" :key="'m'+i" class="sk-card">
          <el-skeleton-item variant="rect" style="width:40px;height:40px" />
          <el-skeleton-item variant="text" style="width:50px" />
        </div>
      </div>
    </div>
    <div v-if="withDatasets" class="sk-section">
      <el-skeleton-item variant="text" style="width:60px;margin-bottom:10px" />
      <div class="sk-card-grid">
        <div v-for="i in 4" :key="'d'+i" class="sk-card">
          <el-skeleton-item variant="rect" style="width:40px;height:40px" />
          <el-skeleton-item variant="text" style="width:70px" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Selection-step loading skeleton shared by the task and search launch pages.
withDefaults(defineProps<{ withDatasets?: boolean }>(), { withDatasets: true })
</script>

<style scoped>
.selection-step-skeleton {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.sk-section {
  display: flex;
  flex-direction: column;
}

.sk-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}

.sk-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 18px 12px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
}
</style>
