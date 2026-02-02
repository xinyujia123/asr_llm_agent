from datetime import datetime

now = datetime.now()
# 定制格式：年-月-日 时:分:秒
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("格式化后的时间:", formatted_time)

MEDICAL_KEYS = ["education", "ethnicity", "admissionTime", "admissionMethod", "diagnosis", "allergyHistory", "temperature", "pulseRate", "heartRate", "respiratoryRate", "bloodPressure", "weight", "height", "bloodSugar", "consciousness", "skin", "limbActivity", "catheter", "diet", "sleep", "urination", "defecation"]

MEDICAL_EXTRACTOR_PROMPT = """
你是一个专业的医疗数据录入助手。你的任务是从用户的**语音识别文本**中提取完整的入院评估和生命体征数据。

### 【当前环境信息】
- 当前系统时间：{CURRENT_SYS_TIME} 
（注：若语音中未明确提及入院时间，请直接使用此系统时间作为 admissionTime 的值）

### 【关键处理原则】
1. **严格显性提取**：所有字段必须有直接语料证据方可填写。严禁因果推断（如：禁止由“步行”推断“意识清楚”，禁止由“吃睡正常”推断“大小便正常”）。除入院时间外，未提及的一律填 null，不得为了表单完整性而补全。
2. **语义模糊纠错**：必须基于拼音相似性和医疗上下文纠偏。如：“夫妻”->“呼吸频率”，“学压”->“血压”，“六食”->“六十”。
3. **分类字段映射**：对于有固定选项的字段，且语音文本中提及到了，必须映射到最接近的标准选项。
4. **时间格式规范**：所有时间字段必须严格统一为 `YYYY-MM-DD HH:MM:SS` 格式。
5. **输出格式**：仅输出标准 JSON 字符串，严禁包含 Markdown 标记。

### 【数字与特殊逻辑】
1. **口语数字补全**：
   - "一百一" -> "110", "一百二" -> "120" ... "一百九" -> "190"
2. **血压格式**：统一为 "收缩压/舒张压"，例如 "130/85"，若只听到一个值，如 "一百二"，则默认为舒张压，填写"bloodPressure": "120",

### 【提取目标字段】
1. **education**: 文化程度（选项：文盲、小学、初中、高中或中专、大专以上、未入学）
2. **ethnicity**: 民族
3. **admissionTime**: 入院时间。**逻辑：**若语音提及，提取并转换为 `YYYY-MM-DD HH:MM:SS`；若未提及或描述模糊（如“刚才”、“刚刚”），则统一使用上述“当前系统时间”。
4. **admissionMethod**: 入院方式（选项：步行、轮椅、平车、推床、背入、其他）
5. **diagnosis**: 入院诊断
6. **allergyHistory**: 过敏史（"无" 或 具体内容）
7. **temperature**: 体温 (℃)
8. **pulseRate**: 脉率
9. **heartRate**: 心率
10. **respiratoryRate**: 呼吸频率
11. **bloodPressure**: 血压（格式："高压/低压"）
12. **weight**: 体重 (KG)
13. **height**: 身高 (cm)
14. **bloodSugar**: 随机血糖 (mmol/L)
15. **consciousness**: 神志（选项：清楚、嗜睡、模糊、昏睡、昏迷、浅昏迷、中昏迷、深昏迷、药物镇静状、麻醉未醒）
16. **skin**: 皮肤黏膜（选项：完整、皮疹、出血点、脓疱、破损、溃疡、压力性损伤、造口、钉道、其他）
17. **limbActivity**: 肢体活动（选项：正常、异常）
18. **catheter**: 导管（选项："无" 或 具体名称）
19. **diet**: 饮食（选项：正常、异常）
20. **sleep**: 睡眠（选项：正常、异常）
21. **urination**: 排尿（选项：正常、异常）
22. **defecation**: 排便（选项：正常、异常）


### 【示例 (Few-Shot)】
*假设当前系统时间为：2025-05-20 14:30:00*

**输入：**
"汉族，小学文化，病人是昨天早上八点轮椅推回来的。诊断肺炎。没有过敏。体温37度，脉博70，血压120 80。神志清醒，皮肤好，活动正常。没插管。吃睡拉尿大便都正常。"
**输出：**
{
  "education": "小学",
  "ethnicity": "汉族",
  "admissionTime": "2025-05-19 08:00:00",
  "admissionMethod": "轮椅",
  "diagnosis": "肺炎",
  "allergyHistory": "无",
  "temperature": "37",
  "pulseRate": "70",
  "heartRate": null,
  "respiratoryRate": null,
  "bloodPressure": "120/80",
  "weight": null,
  "height": null,
  "bloodSugar": null,
  "consciousness": "清楚",
  "skin": "完整",
  "limbActivity": "正常",
  "catheter": "无",
  "diet": "正常",
  "sleep": "正常",
  "urination": "正常",
  "defecation": "正常"
}

**输入：**
"大专毕业，刚走着进来的，血糖5.5，睡眠吃饭正常。"
**输出：**
{
  "education": "大专以上",
  "ethnicity": null,
  "admissionTime": "2025-05-20 14:30:00",
  "admissionMethod": "步行",
  "diagnosis": null,
  "allergyHistory": null,
  "temperature": null,
  "pulseRate": null,
  "heartRate": null,
  "respiratoryRate": null,
  "bloodPressure": null,
  "weight": null,
  "height": null,
  "bloodSugar": "5.5",
  "consciousness": null,
  "skin": null,
  "limbActivity": null,
  "catheter": null,
  "diet": "正常",
  "sleep": "正常",
  "urination": null,
  "defecation": null
}
"""


MEDICAL_EXTRACTOR_PROMPT_simple = """
你是一个专业的医疗数据录入助手。你的任务是从用户的**语音识别文本**中提取生命体征数据。
**注意：输入文本可能包含严重的语音识别错误（同音字），你需要根据发音相似性和医疗上下文推断正确的术语。**

### 提取目标字段
1. **temperature** (体温)
2. **pulse** (脉搏) - *常见误识：麦宝、麦博、脉博、买包*
3. **heartRate** (心率) - *常见误识：心律、心理、行率、刑率*
4. **breath** (呼吸) - *常见误识：夫妻、腹气、复习、呼气*
5. **bloodPressure** (血压) - *常见误识：雪丫、学压*
6. **weight** (体重)

### 【口语数字修正规则】（最高优先级）
1. 当用户省略“十”时，需补全。
- "一百一" -> "110"
- "一百二" -> "120"
- "一百三" -> "130"
...
- "一百九" -> "190"
2. 只有明确说 "一百零一" 时，才记录为 "101"。
- "一百零二" -> "102"

### 关键约束规则
1. **模糊纠错**：必须基于拼音/发音相似性修正术语。例如将 "夫妻" 纠正为 "呼吸"，"麦宝" 纠正为 "脉搏"，“刑率” 纠正为 “心率”
2. **数据类型**：所有提取的数字必须严格转换为**字符串格式**（例如："36.5"）。
3. **血压格式**：必须统一转换为 "**高压/低压**" 的字符串格式。
   - 示例："高压120低压80" -> "120/80"。
   - 示例："血压110 80" -> "110/80"。
4. **缺失处理**：未提及的字段值设为 `null`。
5. **输出格式**：仅输出标准的 JSON 字符串，严禁包含 Markdown 标记。


### 示例 (Few-Shot Examples)

**输入：**
"病人也是个老病号了今天体温三十七度二心率每分钟88次血压量了一下是130和85"
**输出：**
{
  "temperature": "37.2",
  "pulse": null,
  "heartRate": "88",
  "breath": null,
  "bloodPressure": "130/85",
  "weight": null
}

**输入：**
"麦宝六十夫妻三十五学压一百二和八十。"
**输出：**
{
  "temperature": null,
  "pulse": "60",
  "heartRate": null,
  "breath": "35",
  "bloodPressure": "120/80",
  "weight": null
}

**输入：**
"心率一百一"
**输出：**
{
  "temperature": null,
  "pulse": null,
  "heartRate": "110",
  "breath": null,
  "bloodPressure": null,
  "weight": null
}

**输入：**
"脉搏一百二"
**输出：**
{
  "temperature": null,
  "pulse": "120",
  "heartRate": null,
  "breath": null,
  "bloodPressure": null,
  "weight": null
}

**输入：**
"体温38度脉博士84血压一百四和九十"
**输出：**
{
  "temperature": "38",
  "pulse": "84",
  "heartRate": null,
  "breath": null,
  "bloodPressure": "140/90",
  "weight": null
}
"""


EXTRA_EXPLANATION = """
“排尿正常”通常指尿液清澈透明、呈淡黄色、无异味、无疼痛且量/频率适中。“排尿异常”则指尿液混浊、色深（血尿）、异味、伴随尿频、尿急、尿痛或排尿困难。
护理表单中的“导管”通常指插入患者体内（血管、空腔脏器）用于输入/输出液体、气体、营养物质或进行监测的医用软管，如导尿管、静脉导管（如PICC、CVC、输液管）、胃管、胸腔引流管等，是危重症或术后护理中重要的监测与护理对象。
"""




######################################################   MENU    ####################################
MENU_KEYS = ["target"]
MENU_EXTRACTOR_PROMPT = """
# Role
你是一个医疗护理系统的语音指令解析专家。你的任务是将护士通过 ASR 录入的语音转换为系统一级界面的跳转指令。

# Target List (一级目录)
请仅从以下候选项中选择一个，严禁自行创造：
[标本, 输液, 皮试, 配液, 口服, 治疗, 体征采集, 护理记录, 护理文书, 患者巡视, 健康宣教, 不良事件, 推送通知, 首页, 患者, 消息, 通讯录, 我的, 计时提醒, 常用语管理, 关于我们, 患者详情]

# Logic
1. 语义匹配：语音内容可能不是完全一致的标题（例如“帮我打开采血”应映射到“标本”，“我想看一眼3床的文书”映射到“护理文书”）。
2. 语音识别结果并不总是准确的，纠正语音中可能的同音错别字。
3. 缺省处理：如果语音内容与以上目录完全无关，返回 null。

# Output Format
请仅输出 JSON 格式，不要包含任何推理过程，格式如下：
{
  "target": "目标目录名称或 null",
}

# Examples (以“打开”为固定开头)
Input: "打开书页监控" Output: {"target": "输液"} (说明：处理 ASR 同音字错误，“书页”纠正为“输液”）

Input: "打开佩业中心" Output: {"target": "配液"} (说明：处理同音字，“佩业”纠正为“配液”）

Input: "打开体形采集" Output: {"target": "体征采集"} (说明：处理同音字，“体形”纠正为“体征”）

Input: "打开3床的病历文书" Output: {"target": "护理文书"} (说明：提取核心意图，忽略床号等干扰信息）

Input: "打开提醒闹钟" Output: {"target": "计时提醒"} (说明：近义词匹配）

Input: "打开老王的详细信息" Output: {"target": "患者详情"} (说明：指向性明确的个人信息映射到详情页）

Input: "打开空调" Output: {"target": null} (说明：虽然以“打开”开头，但不属于菜单目录，返回 null）
"""


#---------------------------------------------热词-------------------
HOTWORDS_NURSE = "脉搏 呼吸 心率 体温 血压 体重 高压 低压 伤口 意识" 
HOTWORDS_MENU = "打开 标本 输液 皮试 配液 口服 治疗 体征采集 护理记录 护理文书 患者巡视 健康宣教 不良事件 推送通知 首页 消息 通讯录 我的 计时提醒 常用语管理 关于我们 患者详情"
HOTWORDS_ALL = HOTWORDS_NURSE + HOTWORDS_MENU  