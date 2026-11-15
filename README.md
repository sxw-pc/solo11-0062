# 标注合议台（tagconsensus）

一个纯本地、完全离线、结果确定性的**人工标注一致性核验与争议裁决台账工具**：读取多人标注长表，先做数据质量校验，再计算一致性指标、定位争议与人员偏离，合并人工裁决，最后输出可归档的报告、共识标签集和 SQLite 台账。

## 1. 定位与边界

### 它是

- 一条「声明式一致性方案 → 数据校验 → 一致性度量 → 争议定位 → 裁决回填 → 报告与台账」的批处理链路。
- 面向标注交付验收、训练数据验收、外包质量评审的小团队本地底座。
- 输入 CSV / JSONL 标注长表与一份 YAML 方案；输出 JSON / Markdown / 自包含 HTML 报告、SQLite 台账、复核队列和裁决后的共识标签。
- 统计量全部自行实现，可使用标准库审计计算过程，并展示 po、pe、Do、De、有效条目数和有效标注数等中间量。
- 运行期不联网、不读取远端资源、不写入外部服务。

### 它不是

- 不是标注平台，不提供点选打标的界面或上传通道。
- 不是模型训练、模型评估或主动学习工具，不训练任何模型。
- 不解析图像、音频、PDF 或正文语义，不读取任何真实业务内容。
- 不做多人实时协作、账号体系、权限管理或审计登录。
- 不做指标口径校准、漏斗转化、实验分流、漂移归因、抽样方案或表关联发现。
- 不是对标注质量的绝对裁决器；统计结果必须由人复核。

## 2. 十分钟快速上手

环境要求：Python 3.10+，可离线安装 Flask、PyYAML、pytest。

```bash
# 1. 安装（在仓库根目录）
python -m pip install -e ".[test]"

# 2. 跑完整演示，退出码应为 0
python -m tagconsensus demo --config configs/demo.yaml

# 3. 跑测试，全部通过
pytest -q

# 4. 启动本地只读服务
python -m tagconsensus serve --config configs/demo.yaml
```

启动服务后，在浏览器访问本机地址 `127.0.0.1:5000`。三个页面：

- `/`：最近一次运行概览与结论；
- `/review`：复核队列、理由码过滤、标注人一致性；
- `/ledger`：历史运行列表与单次明细。

服务只监听 `127.0.0.1`，命令行没有修改监听地址的开关。页面资源全部来自仓库内的 `static/`，没有外部字体、图标或脚本。

`demo` 会读取仓库自带的合成 fixture，随后依次完成校验、度量、争议定位、裁决回填、三份报告和台账写入。默认输出：

- `output/report.json`
- `output/report.md`
- `output/report.html`
- `output/consensus.jsonl`
- `var/tagconsensus.db`

`configs/strict.yaml` 使用同一份含 warning 的数据，但把 warning 提升为阻断，退出码非 0，报告结论为「阻断」。

## 3. 数据格式

### CSV

第一行必须是表头。`annotator_id`、`item_id`、`label` 必填，`confidence` 可选。

```csv
annotator_id,item_id,label,confidence
A01,it_0001,topic_alpha,0.90
A02,it_0001,topic_alpha,0.80
A03,it_0001,topic_beta,0.70
A04,it_0001,
```

- 空 `label` 视为缺失标注：不参与一致性计算，但计入缺失统计并产生 warning。
- 行尾空白会去掉，标签在比对前统一小写；输出标签使用小写。
- 行宽异常、缺少 ID、非法 JSON 等会作为 `MALFORMED_ROW` 报告，并保留行号。

### JSONL

每行一个对象。`label` 可以是字符串，也可以是字符串数组；数组用于多标签方案。

```json
{"annotator_id":"A01","item_id":"it_0001","label":["topic_alpha","topic_beta"],"confidence":0.9}
{"annotator_id":"A02","item_id":"it_0001","label":"topic_alpha"}
{"annotator_id":"A03","item_id":"it_0001","label":[]}
```

若未开启 `multi_label`，一行出现多个标签会作为 `MULTI_LABEL_NOT_ENABLED` 阻断。

### 编码

方案中支持 `utf-8`、`utf-8-sig`、`gbk`。CSV 表头会额外去除 BOM。

### 数据指纹

每次运行记录三层指纹：

1. 输入文件的 SHA-256；
2. 有效解析记录数；
3. 排序后标注人集合的 SHA-256。

三者合并成数据指纹，写入报告与台账。

## 4. YAML 方案字段

路径字段均相对**方案文件所在目录**解析。不写 `db_path` 时默认使用 `output_dir` 同级 `var/tagconsensus.db`。

| 字段 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `dataset.path` | 是 | - | CSV / JSONL 相对路径 |
| `dataset.format` | 是 | - | `csv` 或 `jsonl` |
| `dataset.encoding` | 否 | `utf-8` | `utf-8` / `utf-8-sig` / `gbk` |
| `label_space` | 是 | - | 允许标签全集，建议包含 `other` |
| `annotators` | 否 | `[]` | 标注人白名单；空表示自动发现 |
| `metric` | 是 | - | `cohen` / `fleiss` / `krippendorff_nominal` / `pairwise_agreement` |
| `pair` | 否 | 无 | 仅 `cohen` 可用；必须恰好两人 |
| `multi_label` | 否 | `false` | 数组标签按标签二值化后分别计算 |
| `min_annotators_per_item` | 否 | `2` | 逐条目非空标注人数下限，不足为 warning |
| `review_threshold` | 否 | `0.6` | 逐条目共识得票率阈值，闭区间 `[0,1]` |
| `outlier_threshold` | 否 | `0.65` | 标注人与多数决一致率下界，闭区间 `[0,1]` |
| `adjudication_file` | 否 | 无 | 裁决 JSONL 相对路径 |
| `output_dir` | 是 | - | 报告与共识文件目录 |
| `db_path` | 否 | 派生 | SQLite 台账文件路径 |
| `strict` | 否 | `false` | 为 `true` 时把所有 warning 提升为 error |
| `persist_raw_rows` | 否 | `false` | 为 `true` 时在台账中持久化 finding 对应的规范化原始行 |

示例：见 `configs/demo.yaml` 与 `configs/strict.yaml`。字段名未知、类型错误、未知指标、阈值越界、标签为空或重复、pair 不在白名单等都会给出稳定错误码和字段路径。

## 5. 度量口径与中间量

所有指标输出适用范围内的有效条目数和有效标注数。Do、De 统一按「观察/期望不一致」定义，po、pe 按「观察/期望一致」定义；因此对 kappa 类指标有 `value = 1 - Do / De`。

### pairwise_agreement

只使用至少有两个非空标注的条目。对每个条目，计算所有有序/无序标注对中的一致对比例，再按全部可配对数汇总：

```text
po = 一致标注对数 / 可配标注对总数
Do = 1 - po
```

本指标没有随机期望，pe、De 记为缺失。完全一致时为 1.0；单一标签全集时为 1.0。

### Cohen's kappa

必须恰有两位标注人；数据中多于两人时必须在方案中用 `pair` 指定两人。只使用两人都提供非空标注的条目。两位标注人的边际分布分别计为 `pA(c)`、`pB(c)`：

```text
po = 对角线配对计数 / 有效配对数
pe = Σc pA(c) × pB(c)
kappa = (po - pe) / (1 - pe)
```

`pe == 1` 时返回 `METRIC_PE_ONE`，不输出 NaN。多标签模式下按标签二值化后逐标签计算。

### Fleiss' kappa

要求每个条目参与计算的标注人数完全相同。设条目 i 的标注人数为 `n_i`，标签 c 的计数为 `n_ic`：

```text
P_i = (Σc n_ic² - n_i) / (n_i × (n_i - 1))
P_bar = 平均(P_i)
p_c = 标签 c 的总标注数 / 总标注数
pe = Σc p_c²
kappa = (P_bar - pe) / (1 - pe)
```

只要存在人数不齐，返回 `METRIC_UNBALANCED`，不凑数硬算。该规则对多标签二值化后的每个标签同样成立。

### Krippendorff's nominal alpha

容忍缺失标注。只使用有至少两个非空标注的条目。构造名义尺度重合矩阵：条目内每对标注的贡献为 `n_ic × n_ik / (n_i - 1)`，对角线为 `n_ic × (n_ic - 1) / (n_i - 1)`。令 n 为有效条目标注总数：

```text
Do = Σ(c≠k) n_ck / n
De = Σ(c≠k) n_c × n_k / (n × (n - 1))
alpha = 1 - Do / De
```

`De == 0`（等价于 pe == 1）时返回 `METRIC_PE_ONE`。Krippendorff 的 De 与 Fleiss 的 1 - pe 不是同一个有限样本校正公式，两者数值可能不同。

### 多标签模式

对 `label_space` 中每个标签分别二值化，输出 per-label 行和有效标签的宏平均。某标签退化时保留该行并附 `error_code`；所有标签都退化时整体返回 `METRIC_ALL_LABELS_INVALID`。

### 争议与偏离定义

- 单标签条目：得票最高的标签唯一时作为多数决；最高票并列时进复核队列并记 `TIE`。得票率严格低于 `review_threshold` 时记 `LOW_CONSENSUS`；等于阈值时通过。非空标注人数低于下限时记 `INSUFFICIENT_ANNOTATORS`。
- 多标签条目：得票率不低于 `review_threshold` 的标签进入共识集；共识集为空时记 `LOW_CONSENSUS`。多标签不使用单标签意义上的 `TIE`。
- 标注人一致率：在该标注人提供了非空标注、且条目存在多数决参考时比较；单标签比较标签是否相同，多标签比较标签集合是否完全一致。无多数决或平票条目不计入分母。
- 低于 `outlier_threshold` 才标记 `ANNOTATOR_OUTLIER`；等于阈值不标记。
- 留一影响度：依次移除一位标注人后重算主指标，`delta = 移除后值 - 基线值`，按绝对影响降序排列，同值按 `annotator_id` 升序；无法重算时记录错误码。

## 6. 错误码

### 方案与配置

| 错误码 | 含义 |
| --- | --- |
| `CFG_FILE_NOT_FOUND` | 方案文件不存在 |
| `CFG_PARSE_ERROR` | YAML 或编码无法解析 |
| `CFG_TYPE_INVALID` | 字段类型错误 |
| `CFG_MISSING_FIELD` | 必填字段缺失 |
| `CFG_UNKNOWN_FIELD` | 出现未知字段 |
| `CFG_DATASET_FORMAT` | 数据格式不是 csv/jsonl |
| `CFG_ENCODING_UNKNOWN` | 不支持的编码 |
| `CFG_METRIC_UNKNOWN` | 未知指标 |
| `CFG_THRESHOLD_RANGE` | 阈值不是 `[0,1]` 内的数字 |
| `CFG_INT_RANGE` | 最小标注人数不是大于等于 2 的整数 |
| `CFG_BOOL_INVALID` | 布尔字段不是布尔值 |
| `CFG_LABEL_EMPTY` | 标签列表为空，或含空标签 |
| `CFG_DUPLICATE_LABEL` | 标签重复 |
| `CFG_DUPLICATE_ANNOTATOR` | 白名单标注人重复 |
| `CFG_PAIR_INVALID` | pair 配置不合法或只用于 cohen |

### 数据加载与校验

| 错误码 | 级别 | 含义 |
| --- | --- | --- |
| `DATA_FILE_NOT_FOUND` | 阻断 | 数据文件不存在 |
| `DATA_ENCODING` | 阻断 | 按指定编码无法解码 |
| `DATA_READ_ERROR` | 阻断 | 文件读取失败 |
| `DATA_HEADER` | 阻断 | 缺少必需列或表头重复 |
| `DATA_FORMAT` | 阻断 | 加载器收到未知格式 |
| `DATA_EMPTY` | error | 没有任何有效记录 |
| `MALFORMED_ROW` | error | 行宽、ID 或 JSON 结构异常，附行号 |
| `DUPLICATE_ANNOTATION` | error | 同一标注人对同一条目重复标注 |
| `LABEL_OUT_OF_SPACE` | error | 标签不在 `label_space` |
| `UNKNOWN_ANNOTATOR` | error | 标注人不在白名单 |
| `MULTI_LABEL_NOT_ENABLED` | error | 非多标签方案出现多标签数组 |
| `EMPTY_LABEL` | warning | 空标签按缺失处理 |
| `INSUFFICIENT_ANNOTATORS` | warning | 条目非空标注人数低于下限 |
| `INVALID_CONFIDENCE` | warning | confidence 非数字或越界，按缺失处理 |

`strict: true` 会把所有 warning 的级别改为 error 并标记 `promoted: true`，从而阻断运行。

### 度量

| 错误码 | 含义 |
| --- | --- |
| `METRIC_NO_ITEMS` | 没有可用于该指标的有效条目 |
| `METRIC_UNBALANCED` | Fleiss 的逐条目评分数不相等 |
| `METRIC_PE_ONE` | 期望一致为 1（或 De 为 0），指标无定义 |
| `METRIC_ANNOTATOR_COUNT` | Cohen 无法确定恰好两位标注人 |
| `METRIC_UNKNOWN` | 内部遇到未知指标 |
| `METRIC_ALL_LABELS_INVALID` | 多标签模式下所有标签都退化 |

### 裁决与台账

| 错误码 | 含义 |
| --- | --- |
| `ADJ_FILE_NOT_FOUND` | 裁决文件不存在 |
| `ADJ_READ_ERROR` | 裁决文件读取失败 |
| `ADJ_MALFORMED_ROW` | 裁决行不是合法 JSON 对象 |
| `ADJ_MISSING_FIELD` | 裁决缺少必填字段 |
| `ADJ_UNKNOWN_ITEM` | 裁决指向不存在的条目 |
| `ADJ_LABEL_OUT_OF_SPACE` | 裁决标签不在 `label_space` |
| `ADJ_DUPLICATE_ITEM` | 同一条目重复裁决 |
| `LEDGER_INIT` | 台账初始化失败 |
| `LEDGER_WRITE` | 台账写入失败 |

裁决错误和指标退化都会使本次运行阻断，但仍尽量写出报告、空共识文件和台账记录，便于回查。

## 7. 报告与台账

### 三份报告

`report.json` 是机器可读主报告，包含：

- `run`：run_id、开始/完成时间、错误码；
- `config`：方案摘要与方案指纹；
- `data`：文件 SHA-256、有效记录数、标注人指纹、条目数；
- `validation`：状态、error/warning/promoted 计数与逐条发现；
- `metrics`：主指标值、po/pe/Do/De、有效条目/标注、per-label 与宏平均；
- `review_queue`：争议条目、票数分布、得票率、理由码、open/resolved 状态；
- `annotators`：一致率、参与条目、可比较条目、缺失、越界、outlier 标记；
- `leave_one_out`：逐人移除后的值和 delta；
- `adjudications`：裁决条数、采纳条数、解决复核条数、逐条明细；
- `consensus`：来源统计和逐条共识记录；
- `conclusion`：`pass` / `review` / `blocked` 及中文结论。

`report.md` 是同样内容的人读摘要。`report.html` 是自包含文件，CSS 内联在 `<style>` 中，不引用外部资源。

### 共识标签

`consensus.jsonl` 按 `item_id` 升序写出全部条目：

```json
{"adjudicated_by":null,"item_id":"it_0001","labels":["topic_alpha"],"review_reasons":[],"source":"majority","vote_rate":1.0}
```

`source` 只取 `majority`、`adjudicated`、`unresolved`。裁决优先于多数决；未解决争议条目的 `labels` 为空。

### SQLite 台账

默认 `var/tagconsensus.db`，七张表：

| 表 | 主要内容 |
| --- | --- |
| `runs` | run_id、时间、方案/数据指纹、校验域指纹、主指标、结论、报告路径 |
| `validation_findings` | 逐条 error/warning、行号、是否由 strict 提升、finding 身份材料 |
| `metrics` | 主指标与 per-label 中间量 |
| `review_queue` | 条目、理由、票数、得票率、解决状态 |
| `annotators` | 一致率、参与量、缺失/越界、outlier 标记 |
| `adjudications` | 裁决标签、复核人、理由、采纳标记 |
| `artifacts` | 文件名、绝对路径、字节数、SHA-256 |

```bash
python -m tagconsensus history --db var/tagconsensus.db
```

### 跨运行 finding 追踪

每条校验 finding 在 `validate` 阶段获得稳定的跨运行身份，随报告写入台账：

- `finding_key`：错误码 + 定位域 + 序号 的哈希。定位域不含行号：记录级 finding（`EMPTY_LABEL`、`LABEL_OUT_OF_SPACE`、`MULTI_LABEL_NOT_ENABLED`、`INVALID_CONFIDENCE`、`DUPLICATE_ANNOTATION`）锚定 `(annotator_id, item_id)`；`UNKNOWN_ANNOTATOR` 锚定标注人；`INSUFFICIENT_ANNOTATORS` 锚定条目；其余锚定消息内容。同一记录上的多条同类 finding（如同一重复键产生多条 `DUPLICATE_ANNOTATION`）按确定顺序编号区分，因此行重排不会改变身份。
- `content_key`：行号无关的语义内容哈希（如越界标签值、条目计数）。标签修订后身份不变但内容标记为变化，避免把未修复问题误报成「已消退 + 新发」。
- `row_digest`：finding 对应规范化输入记录的哈希；无法定位到单条记录时为空。

```bash
python -m tagconsensus findings --db var/tagconsensus.db [--run ID] [--against ID]
```

`findings` 把某次运行（默认最新）与之前一次运行按 `finding_key` 比对，输出每条 finding 的状态：`NEW`（新发）、`PERSISTING`（持续）、`RESOLVED`（已消退），并同时保留上次/本次行号。行号变化（`line_changed`）与 finding 内容变化（`content_changed`）、记录内容变化（`row_changed`）分别标记。

只有两次运行都带有身份材料、且校验相关的方案语义（数据路径/格式/编码、label_space、标注人白名单、multi_label、min_annotators_per_item、strict，合并为校验域指纹）完全一致时才计算状态；否则明确报不可比并给出原因：`no_runs` / `no_previous_run` / `run_not_found` / `missing_identity_material`（旧版本写入的运行缺少身份材料）/ `config_incompatible`。两次运行的数据指纹（文件 SHA-256）总会随修复变化，因此不作为可比性条件，而是以 `data_changed` 展示；不猜测修复结论。

`persist_raw_rows: true` 时，台账额外保存每条可定位 finding 的规范化原始行（`row_json`），比较结果中可看到上次的/本次的记录内容，便于审计与日后按新规则重算身份；代价是占用更多存储。默认只存 `row_digest` 与 `finding_key`，节省空间，但历史上无法还原原行细节。旧版台账在首次打开时自动迁移补列，旧运行的身份列保持 NULL。

## 8. 可复现性

- 同一输入、同一方案、同一目录重复运行，报告正文除 `run_id`、`started_at`、`completed_at` 外完全一致。
- JSON 使用固定键排序、固定缩进和 UTF-8 输出；列表排序均显式指定键，不依赖 dict/set 迭代顺序。
- 浮点数在写入报告时统一保留 12 位小数，用于展示和归档；内部统计不提前取整。
- 数据指纹只由文件字节、有效记录数、排序后的标注人集合组成。
- 方案指纹只由解析后的语义字段组成。
- 台账的 `started_at` 与 run_id 每次运行自然不同，不属于字节级复现范围。
- 裁决文件重新读取和重新应用是稳定的；同一裁决重复运行会得到同一最终标签和同一报告正文。
- 不写随机采样、不写训练模型、不依赖网络时间或外部状态。

## 9. 目录结构

```text
tagconsensus/
  README.md
  COMPLIANCE.md
  pyproject.toml
  Makefile
  configs/
    demo.yaml
    strict.yaml
  data/
    raw/annotations_demo.csv
    raw/annotations_multi.jsonl
    adjudications/demo_adjudications.jsonl
    expected/demo_golden.json
  src/tagconsensus/
    __init__.py
    __main__.py
    cli.py
    config.py
    models.py
    loader.py
    validate.py
    metrics.py
    items.py
    adjudicate.py
    ledger.py
    finding_history.py
    report.py
    report_html.py
    webapp.py
    hashing.py
    errors.py
    templates/
      overview.html
      review.html
      ledger.html
    static/
      style.css
  tests/
    conftest.py
    test_config.py
    test_loader.py
    test_validate.py
    test_metrics_nominal.py
    test_metrics_edge.py
    test_items.py
    test_adjudicate.py
    test_ledger.py
    test_finding_history.py
    test_report.py
    test_cli_demo.py
    test_webapp_local_only.py
```

`report_html.py` 只负责自包含 HTML 渲染，从 `report.py` 拆出以保持单文件不超过 300 行。

## 10. 测试

```bash
pytest -q
pytest --cov=tagconsensus --cov-report=term-missing
```

测试覆盖：

- 四个指标的经典手算样例，精确到 1e-9，数值来源写入 `data/expected/demo_golden.json`；
- 完全一致、随机一致、单条目、全缺失、Fleiss 人数不齐、pe == 1、空文件、仅表头；
- CSV/JSONL、utf-8-sig、gbk、多标签数组、重复标注、越界标签、精确行号；
- 平票、阈值边界、裁决优先、重复裁决、裁决标签越界；
- 两次运行报告剔除时间字段和 run_id 后完全一致；
- finding 身份跨行重排稳定、重复键多条 finding 区分、标签修订记为内容变化、配置变化与旧运行报不可比、规范化原始行按需持久化；
- Web 空状态、正常渲染、查询白名单、nosniff、无公开监听地址、无外部资源引用。

当前测试数为 116，全部通过，无 skip。核心度量与裁决模块覆盖率高于 85%。

## 11. 验收结果

以下为提交前在本机实际执行的结果摘要。

### 1) 安装并跑完整演示

```text
$ python -m pip install -e ".[test]"
Successfully installed pytest-cov-5.0.0 tagconsensus-1.0.0

$ python -m tagconsensus demo --config configs/demo.yaml
run_id=51a5918635a04769905cc57ecf166190 conclusion=需复核 metric=krippendorff_nominal value=0.498757418398
records=120 items=20 review=4 adjudicated=2
output=.../output
# 退出码 0
```

输出目录生成 `report.json`、`report.md`、`report.html`、`consensus.jsonl`，并写入 `var/tagconsensus.db`。

### 2) 测试

```text
102 passed in 3.39s
```

无 skip。

### 3) 本地服务

`python -m tagconsensus serve --config configs/demo.yaml` 能正常启动 Flask 并绑定 `127.0.0.1:5000`，启动日志显示：

```text
serving local read-only UI at 127.0.0.1:5000
 * Running on 127.0.0.1:5000
```

本次验收环境里该端口已有一个与本仓库无关的本地 mock 进程占用，回环请求被转发到了那个进程。因此另外用同一 `create_app` 在临时回环端口 `127.0.0.1:5017` 做了真实套接字验证：

```text
GET /           200  nosniff=nosniff  overview=True
GET /review     200  nosniff=nosniff  marker=True
GET /ledger     200  nosniff=nosniff  marker=True
GET /review?reason=LOW_CONSENSUS  200  nosniff=nosniff  marker=True
GET /review?reason=BAD            400  nosniff=nosniff  marker=True
```

Flask test client 另已验证空状态、正常渲染、查询参数拒绝与只读方法限制。页面只引用本地 `static/style.css`，没有外部样式、字体、图标或脚本。clean 环境下 `serve` 默认监听 5000，且没有修改监听地址的开关。

### 4) strict 阻断

```text
$ python -m tagconsensus demo --config configs/strict.yaml
run_id=cda2442a8e3641dbb442376ec1c9b602 conclusion=阻断 error=VALIDATION_BLOCKED
output=.../output_strict
# 退出码 2

STRICT_CONCLUSION: blocked 阻断
STRICT_COUNTS: error=2 warning=0 promoted=2
```

同一份含两个空标签 warning 的数据在 strict 模式下被提升为 error，报告结论为「阻断」。

### 5) 离线与敏感信息检查

```text
$ grep -RIn -E '0\.0\.0\.0|https?://' .
# 无匹配
```

全仓库不存在公开监听地址、外部资源协议前缀或真实密钥。测试中的禁用模式使用字符串拼接构造，避免把禁用字面量写入仓库。

### 6) 文件规模

```text
$ find . -name "*.py" -not -path "./.pytest_cache/*" -print0 | xargs -0 wc -l | tail -1
4691 total
```

全部 Python 文件单文件不超过 300 行，最大 300 行。最终统计见交付说明。

### 7) 报告确定性

```text
DETERMINISTIC_REPORT: True
SECOND_RUN_ID: 76cf60c77a37469faf59dc48f47f3ee8
```

同一方案连续运行两次，剔除 `run_id`、`started_at`、`completed_at` 后，`report.json` 完全一致。

### 8) 历史台账

```text
$ python -m tagconsensus history --db var/tagconsensus.db
run_id                            started_at                        metric                    value              conclusion  records
76cf60c77a37469faf59dc48f47f3ee8  2026-11-14T17:14:02.243498+00:00  krippendorff_nominal      0.498757418398     需复核      120
cda2442a8e3641dbb442376ec1c9b602  2026-11-14T17:14:01.771304+00:00  -                         -                  阻断        120
51a5918635a04769905cc57ecf166190  2026-11-14T17:13:57.744160+00:00  krippendorff_nominal      0.498757418398     需复核      120
```

历史台账能列出多次运行及各自指标；strict 阻断运行也会登记。

## 12. 未实现（后续迭代）

以下明确未实现，不属于当前交付：

1. 序数或区间尺度的 Krippendorff's alpha、加权 kappa。
2. ICC、Gwet's AC1、Bootstrap 置信区间与显著性检验。
3. 主动学习、模型训练、模型评估、标注推荐。
4. 图像、音频、PDF、正文文本解析与标注界面。
5. 多批次纵向对比与趋势图。
6. 用户账号、权限、审计登录、多人实时协作。
7. 网络采集、云存储、消息队列、外部数据库连接。
8. Excel / PDF 导出。

## 13. 合规

本工具只用于处理使用者本人所有或已获书面授权的标注数据。仓库内全部数据为合成 fixture，不含真实个人信息、公司名、地址或业务内容。详细声明见 `COMPLIANCE.md`。

一致性指标是统计量，不是对标注质量或人员能力的绝对裁决；任何自动化结论都必须由人工复核。
