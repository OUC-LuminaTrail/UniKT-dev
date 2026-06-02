"""模型模块，使用延迟注册实现按需加载。

导入此模块只会注册模型名称，不会加载实际的模型代码。
只有在调用 TRAINERS.get() 或 PARAM_CONFIGS.get() 时才会加载对应模型。
"""

from utils.core import PARAM_CONFIGS, TRAINERS

# =============================================================================
# 训练器延迟注册
# =============================================================================
TRAINERS.register_lazy("BDGKT", "model.BDGKT.BDGKT_trainer", "BDGKTTrainer")
TRAINERS.register_lazy("ABKT", "model.ABKT.ABKT_trainer", "ABKTTrainer")
TRAINERS.register_lazy("AKT", "model.AKT.AKT_trainer", "AKTTrainer")
TRAINERS.register_lazy(
    "ClusterKT", "model.ClusterKT.ClusterKT_trainer", "ClusterKTTrainer"
)
TRAINERS.register_lazy(
    "DTransformer", "model.DTransformer.DTransformer_trainer", "DTransformerTrainer"
)
TRAINERS.register_lazy("DKT", "model.DKT.DKT_trainer", "DKTTrainer")
TRAINERS.register_lazy("DKVMN", "model.DKVMN.DKVMN_trainer", "DKVMNTrainer")
TRAINERS.register_lazy("DyGKT", "model.DyGKT.DyGKT_trainer", "DyGKTTrainer")
TRAINERS.register_lazy("GIKT", "model.GIKT.GIKT_trainer", "GIKTTrainer")
TRAINERS.register_lazy("ReKT", "model.ReKT.ReKT_trainer", "ReKTTrainer")
TRAINERS.register_lazy("RobustKT", "model.RobustKT.RobustKT_trainer", "RobustKTTrainer")
TRAINERS.register_lazy("HawkesKT", "model.HawkesKT.HawkesKT_trainer", "HawkesKTTrainer")
TRAINERS.register_lazy("LBKT", "model.LBKT.LBKT_trainer", "LBKTTrainer")
TRAINERS.register_lazy("GIKTEdmine", "model.GIKTEdmine", "GIKTEdmineTrainer")
TRAINERS.register_lazy("GKT", "model.GKT.GKT_trainer", "GKTTrainer")
TRAINERS.register_lazy("HGIKT", "model.HGIKT.HGIKT_trainer", "HGIKTTrainer")
TRAINERS.register_lazy("SGKT", "model.SGKT.SGKT_trainer", "SGKTTrainer")
TRAINERS.register_lazy("SAKT", "model.SAKT.SAKT_trainer", "SAKTTrainer")
TRAINERS.register_lazy("SimpleKT", "model.SimpleKT.SimpleKT_trainer", "SimpleKTTrainer")
TRAINERS.register_lazy("SQGKT", "model.SQGKT.SQGKT_trainer", "SQGKTTrainer")
TRAINERS.register_lazy("UKT", "model.UKT.UKT_trainer", "UKTTrainer")
TRAINERS.register_lazy("QIKT", "model.QIKT.QIKT_trainer", "QIKTTrainer")
TRAINERS.register_lazy("MCKT", "model.MCKT.MCKT_trainer", "MCKTTrainer")

# HGIKT 变体
TRAINERS.register_lazy(
    "HGIKT_WeightedFusion",
    "model.HGIKT.variants.hgikt_weighted_fusion_trainer",
    "HGIKTWeightedFusionTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_ConcatFusion",
    "model.HGIKT.variants.hgikt_concat_fusion_trainer",
    "HGIKTConcatFusionTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_SimpleFusion",
    "model.HGIKT.variants.hgikt_simple_fusion_trainer",
    "HGIKTSimpleFusionTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_HyperOnly",
    "model.HGIKT.variants.hgikt_hyper_only_trainer",
    "HGIKTHyperOnlyTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_QuestionSkillOnly",
    "model.HGIKT.variants.hgikt_question_skill_only_trainer",
    "HGIKTQuestionSkillOnlyTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_HeteroOnly",
    "model.HGIKT.variants.hgikt_hetero_only_trainer",
    "HGIKTHeteroOnlyTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_HyperOnlySimple",
    "model.HGIKT.variants.hgikt_hyper_only_simple_trainer",
    "HGIKTHyperOnlySimpleTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_Hyper_Only_Unweighted",
    "model.HGIKT.variants.hgikt_hyper_only_unweighted_trainer",
    "HGIKTHyperOnlyUnweightedTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_QS_QT_Only",
    "model.HGIKT.variants.hgikt_qs_qt_only_trainer",
    "HGIKTQSSAOnlyTrainer",
)
TRAINERS.register_lazy(
    "HGIKT_QS_SA_Only",
    "model.HGIKT.variants.hgikt_qs_sa_only_trainer",
    "HGIKTQSSAOnlyTrainer",
)

# =============================================================================
# 参数配置延迟注册
# =============================================================================
PARAM_CONFIGS.register_lazy("BDGKT", "model.BDGKT.BDGKT_trainer", "BDGKTModelParams")
PARAM_CONFIGS.register_lazy("ABKT", "model.ABKT.ABKT_trainer", "ABKTModelParams")
PARAM_CONFIGS.register_lazy("AKT", "model.AKT.AKT_trainer", "AKTModelParams")
PARAM_CONFIGS.register_lazy(
    "ClusterKT", "model.ClusterKT.ClusterKT_trainer", "ClusterKTModelParams"
)
PARAM_CONFIGS.register_lazy(
    "DTransformer", "model.DTransformer.DTransformer_trainer", "DTransformerModelParams"
)
PARAM_CONFIGS.register_lazy("DKT", "model.DKT.DKT_trainer", "DKTModelParams")
PARAM_CONFIGS.register_lazy("DKVMN", "model.DKVMN.DKVMN_trainer", "DKVMNModelParams")
PARAM_CONFIGS.register_lazy("DyGKT", "model.DyGKT.DyGKT_trainer", "DyGKTModelParams")
PARAM_CONFIGS.register_lazy("GIKT", "model.GIKT.GIKT_trainer", "GIKTModelParams")
PARAM_CONFIGS.register_lazy("ReKT", "model.ReKT.ReKT_trainer", "ReKTModelParams")
PARAM_CONFIGS.register_lazy(
    "RobustKT", "model.RobustKT.RobustKT_trainer", "RobustKTModelParams"
)
PARAM_CONFIGS.register_lazy(
    "HawkesKT", "model.HawkesKT.HawkesKT_trainer", "HawkesKTModelParams"
)
PARAM_CONFIGS.register_lazy("LBKT", "model.LBKT.LBKT_trainer", "LBKTModelParams")
PARAM_CONFIGS.register_lazy("GIKTEdmine", "model.GIKTEdmine", "GIKTEdmineModelParams")
PARAM_CONFIGS.register_lazy("GKT", "model.GKT.GKT_trainer", "GKTModelParams")
PARAM_CONFIGS.register_lazy("HGIKT", "model.HGIKT.HGIKT_trainer", "HGIKTModelParams")
PARAM_CONFIGS.register_lazy("SGKT", "model.SGKT.SGKT_trainer", "SGKTModelParams")
PARAM_CONFIGS.register_lazy("SAKT", "model.SAKT.SAKT_trainer", "SAKTModelParams")
PARAM_CONFIGS.register_lazy(
    "SimpleKT", "model.SimpleKT.SimpleKT_trainer", "SimpleKTModelParams"
)
PARAM_CONFIGS.register_lazy("SQGKT", "model.SQGKT.SQGKT_trainer", "SQGKTModelParams")
PARAM_CONFIGS.register_lazy("UKT", "model.UKT.UKT_trainer", "UKTModelParams")
PARAM_CONFIGS.register_lazy("QIKT", "model.QIKT.QIKT_trainer", "QIKTModelParams")
PARAM_CONFIGS.register_lazy("MCKT", "model.MCKT.MCKT_trainer", "MCKTModelParams")

# HGIKT 变体参数配置
PARAM_CONFIGS.register_lazy(
    "HGIKT_WeightedFusion",
    "model.HGIKT.variants.hgikt_weighted_fusion_trainer",
    "HGIKTWeightedFusionModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_ConcatFusion",
    "model.HGIKT.variants.hgikt_concat_fusion_trainer",
    "HGIKTConcatFusionModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_SimpleFusion",
    "model.HGIKT.variants.hgikt_simple_fusion_trainer",
    "HGIKTSimpleFusionModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_HyperOnly",
    "model.HGIKT.variants.hgikt_hyper_only_trainer",
    "HGIKTHyperOnlyModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_QuestionSkillOnly",
    "model.HGIKT.variants.hgikt_question_skill_only_trainer",
    "HGIKTQuestionSkillOnlyModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_HeteroOnly",
    "model.HGIKT.variants.hgikt_hetero_only_trainer",
    "HGIKTHeteroOnlyModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_HyperOnlySimple",
    "model.HGIKT.variants.hgikt_hyper_only_simple_trainer",
    "HGIKTHyperOnlySimpleModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_Hyper_Only_Unweighted",
    "model.HGIKT.variants.hgikt_hyper_only_unweighted_trainer",
    "HGIKTHyperOnlyUnweightedModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_QS_QT_Only",
    "model.HGIKT.variants.hgikt_qs_qt_only_trainer",
    "HGIKTQSSAOnlyModelParams",
)
PARAM_CONFIGS.register_lazy(
    "HGIKT_QS_SA_Only",
    "model.HGIKT.variants.hgikt_qs_sa_only_trainer",
    "HGIKTQSSAOnlyModelParams",
)
