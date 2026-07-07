# 语言覆盖调研报告

数据来源：`CohereForAI/aya_dataset`（train split）、`facebook/belebele`、`Davlan/sib200`  
查询时间：2026-06-15

---

## 一、完整覆盖表（Aya ∩ Belebele ∩ SIB-200）

> 仅列出 Aya 中存在且在 Belebele、SIB-200 均有测试集的语言，按 Aya 条数降序排列。
> ✅ = 有测试集，❌ = 无测试集。

| ISO | 语言 | 字符系统 | Aya 条数 | Belebele | SIB-200 | 备注 |
|---|---|---|---|---|---|---|
| yo | Yoruba（约鲁巴）| Latin | 11758 | ✅ yor_Latn | ✅ yor_Latn | **现有**（Phase 2 仅用 500）|
| so | Somali（索马里）| Latin | 7704 | ✅ som_Latn | ✅ som_Latn | 候选 |
| ha | Hausa（豪萨）| Latin | 3512 | ✅ hau_Latn | ✅ hau_Latn | 候选 |
| zh | Simplified Chinese（中文）| Han | 3038 | ✅ zho_Hans | ✅ zho_Hans | **现有** |
| wo | Wolof（沃洛夫）| Latin | 2914 | ✅ wol_Latn | ✅ wol_Latn | 候选 |
| zu | Zulu（祖鲁）| Latin | 1833 | ✅ zul_Latn | ✅ zul_Latn | 候选 |
| bn | Bengali（孟加拉）| Bengali | 1534 | ✅ ben_Beng | ✅ ben_Beng | **现有** |
| ig | Igbo（伊博）| Latin | 1534 | ✅ ibo_Latn | ✅ ibo_Latn | 候选 |
| fr | French（法语）| Latin | 1422 | ✅ fra_Latn | ✅ fra_Latn | **现有** |
| sn | Shona（绍纳）| Latin | 1368 | ✅ sna_Latn | ✅ sna_Latn | 候选 |
| am | Amharic（阿姆哈拉）| Ge'ez | 1207 | ✅ amh_Ethi | ✅ amh_Ethi | 候选 |
| th | Thai（泰语）| Thai | 724 | ✅ tha_Thai | ✅ tha_Thai | **现有** |
| ny | Nyanja（尼扬贾）| Latin | 688 | ✅ nya_Latn | ✅ nya_Latn | 候选 |
| my | Burmese（缅甸语）| Myanmar | 472 | ✅ mya_Mymr | ✅ mya_Mymr | 候选 |
| xh | Xhosa（科萨）| Latin | 377 | ✅ xho_Latn | ✅ xho_Latn | 候选 |
| sw | Swahili（斯瓦希里）| Latin | 366 | ✅ swh_Latn | ✅ swh_Latn | **现有** |
| nso | Northern Sotho（北索托）| Latin | 141 | ✅ nso_Latn | ✅ nso_Latn | 候选（极低）|

> **注**：Traditional Chinese（1871 条）已在 Phase 2 中合并到 zh 使用（共 4909 条）。

---

## 二、现有语言选择（Phase 2）

| ISO | 语言 | Aya 条数 | 实际用量 | NLLB 补充 | 问题 |
|---|---|---|---|---|---|
| en | English | 3944（Open-Platypus 24926）| 500 | 0 | 无 |
| fr | French | 1422 | 1422（全部）| 3578 | 71% 为机器翻译 |
| zh | Simplified Chinese | 3038+1871=4909 | 4909（全部）| 0 | 中等资源，可考虑移除 |
| sw | Swahili | 366 | 366（全部）| 1634 | 82% 为机器翻译 |
| th | Thai | 724 | 724（全部）| 1276 | 64% 为机器翻译 |
| bn | Bengali | 1534 | 1534（全部）| 466 | 23% 为机器翻译 |
| yo | Yoruba | **11758** | **500（截断！）** | 0 | 严重浪费，是本次 ablation 的核心问题 |

**核心问题**：
- yo 有 11758 条但只用了 500 条（`SAMPLE_SIZES['yo'] = 500` 人为截断）
- sw/th/fr 的 Aya 原生数据极少，大量依赖 NLLB 机器翻译数据填充
- zh 是中等资源语言，与研究"低资源干扰"的核心问题相关性较弱

---

## 三、推荐新语言集

移除 zh（中等资源），yo 使用全量（11758 条），新增三个低资源语言，总计 7 种目标语言 + en：

| ISO | 语言 | Aya 条数 | 字符系统 | 资源级别 | 选择理由 |
|---|---|---|---|---|---|
| en | English | 24926（Open-Platypus）| Latin | 高（基准）| 基准语言 |
| fr | French | 1422 | Latin | 高资源对照 | 验证方法有效性，预期干扰弱 |
| sw | Swahili | 366 | Latin | 低资源 | 现有，非洲 Latin script |
| th | Thai | 724 | Thai | 低资源 | 现有，非 Latin script |
| bn | Bengali | 1534 | Bengali | 低资源 | 现有，South Asian script |
| yo | Yoruba | **11758** | Latin | 低资源 | 现有，全量使用 |
| so | Somali | 7704 | Latin | 低资源 | 新增，数据充足，东非 |
| ha | Hausa | 3512 | Latin | 低资源 | 新增，西非，大语言但 NLP 资源少 |
| am | Amharic | 1207 | **Ge'ez** | 低资源 | 新增，唯一非 Latin/CJK/Indic 字符系统 |

**Script 多样性**：Latin（en/fr/sw/yo/so/ha）、Thai（th）、Bengali（bn）、Ge'ez（am）——移除 zh 后用 am 补充非 Latin 多样性。

---

## 四、数据质量说明

Phase 2 中各语言训练数据的 Aya 原生比例：

| 语言 | Aya 原生 | NLLB 机翻 | 原生占比 |
|---|---|---|---|
| yo | 500 | 0 | 100% |
| bn | 1534 | 466 | 77% |
| zh | 4909 | 0 | 100% |
| fr | 1422 | 3578 | 28% |
| th | 724 | 1276 | 36% |
| sw | 366 | 1634 | 18% |

**新方案下各语言将 100% 使用 Aya 原生数据**（sw/th/bn 使用全部可用量，yo/so/ha 数据充足）。

---

## 五、Belebele 和 SIB-200 完整配置名对照

### Belebele（共 122 个 config）
已覆盖上表全部语言：`yor_Latn`, `som_Latn`, `hau_Latn`, `zho_Hans`, `wol_Latn`, `zul_Latn`, `ben_Beng`, `ibo_Latn`, `fra_Latn`, `sna_Latn`, `amh_Ethi`, `tha_Thai`, `nya_Latn`, `mya_Mymr`, `xho_Latn`, `swh_Latn`, `nso_Latn`, `eng_Latn`

### SIB-200（共 205 个 config）
已覆盖上表全部语言，config 名与 Belebele 一致（使用相同 FLORES-200 语言代码体系）。
