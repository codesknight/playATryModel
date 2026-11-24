# 从零实现微型文字生成模型：学习笔记

## 学习方式

- 老师每次讲一个小目标，提供少量代码，解释代码、数学原理和验证方法。
- 学习者亲自输入代码、准备数据并运行，将实际代码、输出或报错发回。
- 检查通过后再进入下一步；发现问题先解释原因，再由学习者修正。
- 此文档由老师随学习进度更新，记录真实结果、疑问和纠错，不提前记为完成。
- 新的练习代码放在 scripts/learn/。最初建议练习数据放在 docs/data/；学习者现采用 data/corpus/ 存放采集的小说文本，后续按实际目录讲解。

## 学习路线

1. 数据采集：明确生成目标、文本来源、授权和保存方式。
2. 数据预处理：处理编码、空行、重复、异常字符，保留原始数据。
3. 数据划分与训练格式：先划分独立文本，再建立训练词表、字符编号、输入与标签，避免数据泄漏。
4. 模型架构：逐块实现字符表示、隐藏状态和输出层，理解数组形状、矩阵运算和参数量。
5. 学习原理：概率、softmax、交叉熵、链式法则、时间反向传播和数值梯度检查。
6. 模型训练：小批量、学习率、梯度裁剪、Adam、训练日志与权重保存。
7. 加载与测评：验证重载结果、验证损失、困惑度、生成质量和测试集使用规则。
8. 优化实验：数据质量、过拟合、模型大小、生成温度，以及速度、存储和训练内存的区别。

## 当前基线：已有教学示例

这部分是老师之前完成的示例，不代表学习者已经掌握。

- 模型：NumPy 字符级 RNN，隐藏层大小 64。
- 示例语料：778 个字符，271 个不同字符。
- 参数总数：39,119。
- 已保存权重文件：147,246 字节。
- 示例主要复现和组合训练文本，尚无独立验证集或测试集。

## 第 1 课：采集原始文本，检查数据

状态：基础检查通过。学习者已修正路径、重新采集语料，并完成四文件读取；结果见本课末尾。

### 目标

理解文本是如何进入程序的，能区分字符总数与不同字符数。

### 数据要求

- 新建 docs/data/raw.txt，保存为 UTF-8。
- 自己写 20～30 条同一风格的中文短文，每条占一行，每条约 30～60 字。
- 推荐围绕“小猫在村庄里的日常”；标点自然、内容有变化，不靠复制句子凑数量。
- 换行表示一条文本的边界。后续划分数据时按整条文本划分，避免把同一条的相邻片段分到训练集和验证集。
- 原始文件保留不动，后续清洗生成新文件。
- 这批小数据用于理解流程，不足以训练通用聊天模型。

### 第一个代码块

由学习者创建 scripts/learn/01_inspect_data.py，逐行输入：

```python
from pathlib import Path  # 导入路径工具，用来找到数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 从当前文件向上找到项目根目录。
data_path = ROOT / "docs" / "data" / "raw.txt"  # 拼出原始文本文件的位置。
text = data_path.read_text(encoding="utf-8")  # 用 UTF-8 把整个文件读成字符串。
records = [line for line in text.splitlines() if line.strip()]  # 按行拆分，先只排除空白行。
characters = sorted(set(text))  # 去掉重复字符并排序，看看文件中出现过哪些字符。

print("数据路径：", data_path)  # 确认程序读取的是预期文件。
print("字符总数：", len(text))  # 包含标点、空格和换行，统计字符串长度。
print("不同字符数：", len(characters))  # 同一个字出现多次，在这里也只算一次。
print("非空文本条数：", len(records))  # 检查收集了多少条文本。
print("前 100 个字符：", repr(text[:100]))  # 用 repr 显示换行等不易看见的字符。
```

### 本课概念

- `str`：Python 中表示文字的数据类型。
- UTF-8：文字与文件字节之间的编码规则；文件字节数不等于字符数。
- N = len(text)：字符总数，包含重复字符。
- V = len(set(text))：不同字符数，当前仅用于检查数据，不是最终训练词表。
- 对“猫看猫。”，N = 4，V = 3。
- `set` 去掉重复字符，但不会删除原字符串里的任何内容。
- 后续正式词表只从训练集建立，验证集中的未见字符需要单独处理。

### 待检查

1. 学习者编写的完整脚本。
2. 终端实际输出，或完整报错。
3. 原始文本前 3 条。
4. 学习者对字符总数、不同字符数与换行计数的解释。

### 实操结果与纠错

学习者已编写 scripts/learn/01_inspect_data.py，并尝试读取 data/corpus/三国演义.txt，当前尚未通过数据检查。

#### 路径错误

- 实际代码使用 `Path(__file__).resolve().parents[3]`，得到 D:/projects。
- 脚本位于 D:/projects/playatry/scripts/learn/，因此项目根目录应使用 `parents[2]`。
- `parents` 从 0 开始：0 为 learn，1 为 scripts，2 为 playatry，3 为 projects。
- 原先的 docs/data/raw.txt 是教学示例位置，实际使用 data/corpus/ 后应同步修改数据路径。

#### 老师实际检查的语料

四份文件均可用 UTF-8 解码。以下字符数按 Python read_text 读取并统一换行后的字符串统计。

| 文件名 | 开头标示的书名 | 字符数 | 检查结果 |
| --- | --- | ---: | --- |
| 三国演义.txt | 三国演义 | 604,703 | 名称与开头一致 |
| 水浒传.txt | 三国演义 | 604,703 | 与三国演义.txt 的 SHA-256 完全相同 |
| 红楼梦.txt | 水浒传 | 885,266 | 名称与开头不符 |
| 西游记.txt | 红楼梦 | 855,342 | 名称与开头不符 |

结论：当前只有三份不同的文件内容，尚未找到开头标示为西游记的语料。开头检查不能代替全文完整性检查。

#### 下一步实操

1. 由学习者核对采集来源、文件内容和命名，保留原始下载备份，补齐真正的西游记文本。
2. 将 ROOT 改为 parents[2]，确认 corpus_dir 指向项目内的 data/corpus/。
3. 用循环读取四份指定文本，每本单独保存在字典中，打印每本统计与开头，核对身份。
4. 暂不把四本直接拼接，也不修改训练入口；先确认数据，再学习清洗、划分与训练样本制作。

以上问题已在本轮复查中修正，详见下文。老师未代写练习脚本，未修改原始语料。

#### 重新采集后的复查：通过基础检查

老师实际运行学习者的 01_inspect_data.py，程序正常结束。ROOT 已修正为 parents[2]，水浒文件名已改为水浒全传.txt。

| 文件名 | 字符总数 | 不同字符数 | 非空行数 |
| --- | ---: | ---: | ---: |
| 三国演义.txt | 599,460 | 3,912 | 3,236 |
| 水浒全传.txt | 900,163 | 4,278 | 4,402 |
| 红楼梦.txt | 865,713 | 4,344 | 3,072 |
| 西游记.txt | 720,681 | 4,367 | 3,583 |

- 合计字符数：3,086,017；字符集合并集大小：5,675，不能把各书的不同字符数直接相加。
- 四份文件的 SHA-256 均不同，之前整文件重复的问题已解决。
- 均可按 UTF-8 解码，未发现 U+FFFD 替换字符或 NUL 字符；这不等于全文不存在错字或缺漏。
- 开头内容与书名对应；章节样式的行可能同时包含标题和正文，后续划分章节时必须注意，不能直接删掉整行。
- 来源线索记录在 docs/myself_note.md；本次仅检查本地文件，未核验上游许可及版本完整性。

## 第 2 课：温和清洗并保存新文件

状态：实操代码与数据检查通过；学习者已完成 scripts/learn/02_clean_data.py，老师实际运行并复查了四本书的输出。

### 本课规则

1. 从 data/corpus/ 读取，原布置写入 data/processed/；学习者实际采用 data/corpus_processed/，此命名有效，后续按实际目录处理，不覆盖原始文件。
2. 清除每行首尾空白，去掉空白行，剩余各行之间保留一个换行。
3. 保留正文、标点、章节标题和行内空格；本课不做繁简转换、不删重段落、不拼接书籍。
4. 先用小字符串理解函数，再批量处理四本书。

### 核心函数：由学习者输入

```python
def clean_text(text):  # 定义清洗函数，接收原文，返回清洗后的字符串。
    cleaned_lines = []  # 保存处理后仍有内容的行。
    for line in text.splitlines():  # 将原文按行拆开，逐行处理。
        line = line.strip()  # 去掉行首行尾的空白，保留行内空白。
        if not line:  # 如果这一行去掉空白后没有内容。
            continue  # 跳过空行，开始处理下一行。
        cleaned_lines.append(line)  # 将有内容的行放入列表。
    return "\n".join(cleaned_lines)  # 在各行之间加入一个换行，组成新字符串。
```

输入 `"  小猫。 \n\n　小狗。　\n"`，预期得到 `"小猫。\n小狗。"`。strip 只处理每行两端的空白，不处理行内空格或标点。

### 验证方法

- 用 repr 查看小样例的空格和换行变化。
- 反复执行同一清洗，应满足 clean_text(cleaned) == cleaned，即幂等性。
- 输出字符数不应大于输入字符数；检查本规则没有擅自增加内容。
- 查看原始目录仍有四份文件，新目录生成四份清洗文件。
- 逐本检查清洗文件开头，确认没有误删标题或正文。

### 老师预先核对的参考值

以下为当前本地语料按上述规则计算的预期值，并非学习者已完成清洗的结果。统计的是内存中的字符串长度，不是文件字节数。

| 文件名 | 清洗前字符数 | 预期清洗后字符数 | 减少字符数 |
| --- | ---: | ---: | ---: |
| 三国演义.txt | 599,460 | 597,979 | 1,481 |
| 水浒全传.txt | 900,163 | 897,427 | 2,736 |
| 红楼梦.txt | 865,713 | 864,306 | 1,407 |
| 西游记.txt | 720,681 | 718,733 | 1,948 |

### 效率概念

N 表示输入字符数。逐行扫描、strip 和 join 合计需要与 N 大致成正比的工作量，记为 O(N)。本实现将原文、行列表及结果放在内存中，额外空间也为 O(N)。这是增长趋势，不代表每个字符只占一个字节，也不代表准确运行秒数。

### 待提交

学习者的完整清洗脚本、小样例输出、四本书的清洗前后统计，以及“为什么不能直接删除章节标题所在整行”的解释。

### 实际完成与老师复查

- 学习者按规则实现了 clean_text，四份输出均与上表参考字符数一致。
- 老师实际运行完整脚本，结果为总字符数从 3,086,017 降至 3,078,445，共减少 7,572 个字符。
- 四份原始文件的 SHA-256 与第一课检查结果一致，证明这期间原始文件内容没有变化。
- 逐本比较实际输出与独立构造的预期字符串，完全相同；没有空白行，每行首尾无多余空白。
- 去除所有空白后，清洗前后内容逐字符相同，确认本规则没有删除或改变非空白内容。
- 幂等性检查通过。老师另外单独调用学习者的函数，验证了小猫/小狗样例、行内空格保留、纯空白输入返回空字符串。
- 学习者将小样例代码注释保留，这不影响批量处理；老师没有代写或修改第二课脚本。
- 本地输出使用 CRLF 换行，因此清洗文件字节数可能比原文件还大；这与按 Python 字符串统计的字符减少不矛盾，Git 仓库按配置统一为 LF。
- 学习者已将原有模型示例移动至 scripts/example/。老师同步修正示例脚本的根目录计算、文档命令和 CI 路径，避免克隆后示例失效。
- 数据来源和复现方法补充在 data/README.md，GitHub 自动检查增加了第一课运行和第二课清洗结果复现检查。

学习者回答：“清洗时保留行间换行是为了保留部分格式，不然模型学不会何时换行。”核心理解正确。补充：换行也是字符，只有保留相关样本，模型才能从中学习换行规律；它还保留部分段落和标题边界，删除可能造成文本粘连。“保留”不保证模型一定学好；第二课已删除空白行，因此没有完整保留原书排版。行内空格还可能承担分隔意义，不应无差别删除。

## 第 3 课：识别章节边界，分块实现章节切分

状态：章节识别和整章切分的实操检查均通过，460 个非空章节可原样还原四份清洗文本。章节仍在内存中，尚未输出章节文件。

### 为什么先识别章节

后面将以完整章节作为数据划分的单位，再在每个集合内部生成训练窗口。若先制作大量相互重叠的窗口再随机划分，相邻窗口可能落入不同集合，验证结果可能偏乐观。按章划分仍不保证没有重复文本或相似内容，后续还要检查集合间重复。

### 真实数据里的误匹配

红楼梦第 98 行以“第四回中，既将薛家母子……”开头。只匹配行首“第若干回”会把这段正文当成章节，得到 121 个候选。

本批语料真正的回目在“回”后面有普通空格、制表符或全角空格，因此本课要求匹配这个分隔符。此规则针对当前文件格式，不是适用于所有小说的通用规则。

### 第一个代码块：由学习者输入到 scripts/learn/03_split_chapters.py

```python
import re  # 导入正则表达式工具，用规则查找章节开头。
from pathlib import Path  # 导入路径工具，定位清洗后的文本。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
input_dir = ROOT / "data" / "corpus_processed"  # 本课读取第二课保存的清洗结果。
book_names = ["三国演义.txt", "水浒全传.txt", "红楼梦.txt", "西游记.txt"]  # 明确本次检查的四本书。
pattern = re.compile(r"^第[一二三四五六七八九十百零〇两0-9]+回[ \t\u3000]+")  # 匹配行首回目，并要求“回”后有横向空白。

for book_name in book_names:  # 依次检查每本书。
    text = (input_dir / book_name).read_text(encoding="utf-8")  # 读取这本书的清洗结果。
    headings = []  # 保存疑似章节开头的行号和原始行内容。
    for line_number, line in enumerate(text.splitlines(), start=1):  # 逐行读取，同时从 1 开始记录行号。
        if pattern.match(line):  # 判断当前行是否符合回目格式。
            headings.append((line_number, line))  # 保存行号与完整行，不截掉同一行中的正文。
    print("\n书名：", book_name)  # 显示当前检查的文件名。
    print("章节候选数：", len(headings))  # 显示规则找到多少个章节开头。
    assert headings, f"没有找到章节：{book_name}"  # 如果一个候选都没有，就停止并提示检查格式。
    print("第一个候选：", headings[0][0], repr(headings[0][1][:80]))  # 显示第一个候选行号及开头，仅截短显示内容。
    print("最后一个候选：", headings[-1][0], repr(headings[-1][1][:80]))  # 显示最后一个候选，检查末章是否识别。
```

### 正则符号

- `r"..."` 为原始字符串，方便把反斜杠交给正则工具解析。
- `^` 为字符串开头；本代码逐行调用 match，因此对应每行开头。
- `[一二…0-9]` 表示从列出的中文数字或阿拉伯数字中匹配一个字符，后面的 `+` 表示出现一次或更多次。
- `[ \t\u3000]+` 表示至少一个普通空格、制表符或全角空格；刻意不使用会匹配换行的 `\s`。
- 当前规则没有把中文数字转换为数值，也不能仅凭数量证明章节顺序、编号和内容都正确。

### 预期检查结果

三国演义 120 个、水浒全传 120 个、红楼梦 120 个、西游记 100 个，共 460 个候选。学习者已提交完整运行输出，老师复查实际脚本并重新运行，结果一致。

下一块再学习：根据两个相邻章节起点截取整章，保留标题和正文，并验证切分后能够还原原文。本块暂不制作词表、训练窗口或划分数据集。

### 第一块口述题复盘

学习者回答：“一是标题完整性、二是防止‘回’后面的标题文字污染正文或者造成切割。”

前半部分正确，后半部分需要修正：标题不是天然的污染，当前目标是保留完整章节，标题可以作为文本的一部分。更关键的是本批语料有标题和正文在同一行的情况，删除整行或只保存“第几回”会丢失内容。识别边界不等于删除命中的内容。是否单独分离标题，应由后续任务决定，并使用经过验证的规则。

### 第二个代码块：整章切分（等待学习者实现）

先在最外层 for book_name 循环前添加：

```python
book_chapters = {}  # 保存每本书切分出来的章节列表，暂时只放在内存里。
```

在每本书首末候选打印之后，仍位于 for book_name 循环内部，添加：

```python
    lines = text.splitlines(keepends=True)  # 按行拆分并保留每行末尾的换行，方便原样还原。
    assert headings[0][0] == 1, "第一章之前还有内容，需要另行处理"  # 避免静默丢弃序言或其他开头文字。
    chapters = []  # 保存当前这本书的完整章节。
    for index, (line_number, _) in enumerate(headings):  # 依次取出候选的序号和行号，暂时不用候选行内容。
        start = line_number - 1  # 将从 1 开始的行号转换为从 0 开始的列表下标。
        if index + 1 < len(headings):  # 判断当前章节后面是否还有下一章。
            end = headings[index + 1][0] - 1  # 下一章的起点就是当前章的终点，切片不包含这个终点。
        else:  # 当前已经是最后一章，没有下一个章节起点。
            end = len(lines)  # 最后一章一直取到文件末尾。
        chapter_text = "".join(lines[start:end])  # 连接属于当前章节的行，不额外添加或删除换行。
        chapters.append(chapter_text)  # 将完整章节放入列表。
    assert "".join(chapters) == text, "切分后无法还原原文"  # 检查切分没有丢字、重复添加或改变顺序。
    book_chapters[book_name] = chapters  # 保存当前这本书的全部章节。
    print("切分章节数：", len(chapters))  # 显示实际生成了多少个章节字符串。
    print("首章字符数：", len(chapters[0]))  # 显示第一章大小，用于核对边界。
    print("末章字符数：", len(chapters[-1]))  # 检查最后一章也取到了文件结尾。
    print("还原检查：通过")  # 上面的断言通过后，才能执行到这里。
```

### 切片原理与效率

- 行号从 1 开始，列表下标从 0 开始；第 29 行对应下标 28。
- 若第一章从第 1 行开始、第二章从第 29 行开始，第一章应取 lines[0:28]，包括下标 0～27，共 28 行。
- 以起点 s_i 表示第 i 章，章节范围为左闭右开区间 [s_i, s_(i+1))，最后一章的右端点是总行数。
- keepends=True 保留原有换行，因此连接时使用空字符串；再用换行连接会重复加入换行。
- 还原一致证明保全了当前字符串，不能单独证明每个章节边界在语义上正确。
- 设 N 为字符数、K 为章节数，拆行和所有章节的拼接合计为 O(N + K)，内存开销为 O(N + K)，不表示额外内存只等于原文件大小。

### 老师预先验证的参考值

以下是老师用相同切分方法预先核对的结果，不代表学习者已完成第二块。

| 文件 | 章节数 | 首章字符数 | 末章字符数 |
| --- | ---: | ---: | ---: |
| 三国演义.txt | 120 | 4,638 | 6,571 |
| 水浒全传.txt | 120 | 5,566 | 9,107 |
| 红楼梦.txt | 120 | 7,324 | 7,854 |
| 西游记.txt | 100 | 6,926 | 6,088 |

参考值以本轮实际运行输出为准，下一块完成后再记录学生实测结果。

待提交：修改后的脚本、四本书的切分统计与还原结果，以及“keepends=True 后为什么用空字符串连接”的解释。尚不保存章节文件，不建立词表，不划分数据集。

### 第二块实操纠错：字典未初始化

学习者运行到 `book_chapters[book_name] = chapters` 时出现 `NameError: name 'book_chapters' is not defined`。老师检查实际脚本，确认漏掉了最外层书籍循环前的 `book_chapters = {}`。

原理：给字典中的某个键赋值，要求字典对象已经存在。`book_chapters = {}` 创建空字典并绑定变量名；`book_chapters[book_name] = chapters` 向已有字典保存一项，不会自动创建字典。

修正位置：放在 pattern 定义之后、最外层 `for book_name in book_names:` 之前，顶格书写。放在循环内部会每处理一本书就清空之前保存的结果。`chapters = []` 则应放在书籍循环内部，因为每本书需要独立的章节列表。

该问题已由学习者亲手修正，最新复查结果见下文。

### 第三课最终实操复查：通过

学习者已提交四本书全部运行结果。老师检查实际脚本，确认 book_chapters 在最外层书籍循环前初始化，chapters 在书籍循环内分别创建。老师重新运行脚本，并检查最终字典：

- 四本书均保留在 book_chapters 中，没有被后一本覆盖清空。
- 章节数分别为 120、120、120、100，合计 460，所有章节均非空。
- 首章与末章字符数与上表参考值逐项一致。
- 对每本书执行 `"".join(chapters)`，结果与对应清洗文本逐字符相同。
- 当前仅实现识别与无损切分，还原正确不能单独证明所有章节编号和语义边界正确；编号顺序检查可继续完善。

本课已实践：正则表达式、列表和元组、字典初始化、循环内外变量位置、左闭右开切片、保留换行以及还原断言。

关于 keepends=True 后为什么使用空字符串连接的口述题，学习者明确表示不会。老师补充讲解：每行已经带有原来的换行，空字符串连接不会新增字符；再使用换行连接会额外插入换行。目前记录为已讲解，待学习者通过例子确认理解，不记为独立掌握。

下一课计划先将章节保存为可重新读取的结构化记录，保留书名、章节标识与完整文本，验证保存后重载一致，再进行可复现的数据集划分。尚未生成训练、验证或测试集。

## 第 4 课：保存 JSONL 并验证重新读取

状态：实操检查通过。学习者已在 03_split_chapters.py 完成 JSONL 保存与重载验证；口述题需补充解释，见本课末尾。

### 换行的补充讲解

`"甲\n乙".splitlines(keepends=True)` 得到 `["甲\n", "乙"]`。

- 用空字符串连接得到 `"甲\n乙"`，与原文一致。
- 用换行连接得到 `"甲\n\n乙"`，在已有换行之外又插入了一个换行。
- join 的规则是在相邻元素之间插入分隔字符串；它不会自动判断原来有没有分隔符。

### 保存目标与记录格式

在现有 scripts/learn/03_split_chapters.py 末尾增加章节保存和重载验证，输出 data/chapters/chapters.jsonl。本节继续使用现有内存结果，不引入跨脚本导入问题；后续再重构函数和入口。

每个章节记录包含：

- id：书名加章节顺序号，如 三国演义:001，方便后续检查重复或集合交集。
- book：原始文件名，保留来源关系。
- chapter_index：从 1 开始的切分顺序号，不声称已经校验过标题中的真实回数。
- text：完整章节，包含标题、正文和原有换行。

JSONL 每一物理行是一条 JSON 记录。正文换行在文件中写成转义序列，json.loads 后还原成真正换行，因此多段正文仍可保存为一条记录。不要使用 indent 将一条记录展开为多个物理行。

### 实操代码：由学习者输入

在脚本顶部增加：

```python
import json  # 导入 JSON 工具，将字典转成文件文字，再从文字还原字典。
```

现有最外层书籍循环结束后，顶格添加：

```python
records = []  # 创建列表，用来保存四本书的全部章节记录。
for book_name, chapters in book_chapters.items():  # 同时取出每本书的文件名和章节列表。
    for chapter_index, chapter_text in enumerate(chapters, start=1):  # 从 1 开始给每本书的章节编号。
        record = {  # 把章节内容及其来源信息放进一个字典。
            "id": f"{Path(book_name).stem}:{chapter_index:03d}",  # 用书名和至少三位的顺序号组成标识。
            "book": book_name,  # 保留来源文件名。
            "chapter_index": chapter_index,  # 保存整数形式的切分顺序号。
            "text": chapter_text,  # 保存完整章节，不做额外清洗。
        }  # 结束当前章节记录的定义。
        records.append(record)  # 将当前章节记录加入总列表。

output_dir = ROOT / "data" / "chapters"  # 指定章节记录的保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已存在时继续使用。
output_path = output_dir / "chapters.jsonl"  # 指定 JSONL 文件名。
with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 以 UTF-8 和 LF 写入，覆盖同名旧结果，结束后自动关闭文件。
    for record in records:  # 逐条取出章节字典。
        line = json.dumps(record, ensure_ascii=False)  # 将字典转换为 JSON 字符串，让中文保持可读。
        file.write(line + "\n")  # 每条记录写成一行，末尾加一个真正的换行作为记录分隔。
```

然后在文件末尾继续顶格添加：

```python
loaded_records = []  # 创建列表，保存重新读取的记录。
with output_path.open("r", encoding="utf-8") as file:  # 打开刚保存的文件，结束后自动关闭。
    for line in file:  # 每次读取一个物理行，也就是一条 JSON 记录。
        loaded_records.append(json.loads(line))  # 将 JSON 文字还原为字典，并加入列表。
assert loaded_records == records, "保存后重新读取的内容不一致"  # 检查记录顺序、字段及完整正文都保持一致。
assert len({record["id"] for record in loaded_records}) == len(loaded_records), "章节标识重复"  # 检查每条记录的标识都唯一。
print("\n保存路径：", output_path)  # 显示生成的文件位置。
print("记录数：", len(loaded_records))  # 当前四本书应共计 460 条记录。
print("第一条标识：", loaded_records[0]["id"])  # 按当前书籍顺序，应为三国演义:001。
print("第一条字符数：", len(loaded_records[0]["text"]))  # 按当前语料，应为 4638。
print("保存后重载检查：通过")  # 上述断言均通过后才能显示。
```

### 概念与效率

- 序列化是把内存数据转换成可保存的表示；反序列化是读取后还原数据。
- ensure_ascii=False 控制中文的显示方式，不是换行转义开关。
- JSONL 的物理行分隔与正文里的逻辑换行承担不同作用。
- 数据在运行结束后仍保存在磁盘，不需要靠内存变量跨程序运行保存。
- 设章节文本总字符数为 N、记录数为 K，本课序列化和重载的时间、额外内存总体为 O(N + K)；实际 JSON 字节数还包含 UTF-8 编码、字段、标识和转义开销。
- 本节只验证保存与重载，不划分训练集，也没有训练模型。

### 待检查

完整脚本、最后的保存与重载输出，以及对“正文有换行，为什么一章仍能保存为 JSONL 的一行”的解释。完成后更新真实结果、复现说明与自动检查，并上传 GitHub。

### 第四课实操复查：通过

学习者提交的末尾输出：460 条记录，第一条 ID 为 三国演义:001，第一条字符数 4,638，保存后重载检查通过。

老师检查实际脚本并重新运行，另行读取生成文件，确认：

- data/chapters/chapters.jsonl 共 460 个物理行，大小 9,258,686 字节，UTF-8 编码、LF 换行。
- 每行均能单独解析为 JSON，字段为 id、book、chapter_index、text，460 个 ID 唯一。
- 各书顺序号从 1 连续编号，ID 与对应书名及顺序号一致。
- 重载后的每章文字与内存切分结果相同，四本书分别按顺序连接后与清洗文本逐字符一致。
- 第一条 JSON 行中存在反斜杠 n 的转义表示，而解析后的 text 中存在真实换行，证明正文换行未丢失。
- GitHub 自动检查现覆盖章节 JSONL 重新生成后的差异检查。

### 口述题复盘：设计目标与实现机制

学习者回答：“因为是一个章节不能拆分。”这描述了本课希望把整章保存为一条记录的目标，未解释为什么正文中的换行不破坏 JSONL 的物理行结构。

补充：章节不是技术上不能拆分；后续还会在章节内部制作训练窗口。本课选择一章一条记录，是为了保留章节身份，方便先划分数据集。实现一行存储的机制是 json.dumps 将字符串里的真正换行编码成反斜杠加 n，json.loads 再将其解码为真正换行。只有 file.write 在记录末尾添加的实际 LF 用于分隔 JSONL 记录；本课记录内的换行均被转义。

当前记录为代码实操通过、转义原理已补充讲解，不能根据运行成功推断概念已独立掌握。下一课计划按整章进行可复现的数据集划分，再在各集合内制作训练窗口，避免窗口跨集合及跨章节。

## 第 5 课：按完整章节划分数据集并复现

状态：代码实操与复现检查通过，学习者已完成 scripts/learn/04_split_dataset.py 和四份输出文件。口述题及 shuffle 原理补充见本课末尾。

### 集合用途与评估范围

- train：用于计算训练误差并更新模型参数。
- validation：用于选择学习率、模型大小、训练停止时机等；不用于梯度更新。
- test：方案确定后的最终评估，不反复根据其结果调参。本课可以检查记录数、完整性和交集等数据结构，不将测试集用于模型选择。
- 本课每本书分别按章节数约 80%/10%/10% 划分，保证每个集合都有四本书的章节；不是按字符数等分。
- 这是同一批书中留出章节的评估，不表示对未见书籍或其他题材的泛化表现。
- 按章划分能防止同一章的重叠训练窗口跨集合，但不同章节仍可能有重复片段，不能因此声称消除了所有泄漏。

### 可复现规则

固定输入文件、排序规则、算法、随机种子以及兼容的 Python 随机实现。先按书名排序，再按章序号和 ID 排序，使用局部 random.Random(42) 打乱每本书的新列表。保存成员列表的 JSONL 是固定划分的实际依据，不只依赖种子。

原始记录列表和每条记录内容不修改。输出 data/splits/train.jsonl、validation.jsonl、test.jsonl，以及 manifest.json，记录输入文件 SHA-256、种子、算法和输出文件统计及哈希。修改语料后，即使种子不变也可能得到不同划分；进入正式对比实验后应冻结划分。

### 第一块：读取与分组

```python
import hashlib  # 导入哈希工具，为输入和输出文件计算内容指纹。
import json  # 导入 JSON 工具，读取和保存章节记录。
import random  # 导入随机工具，按固定种子打乱章节顺序。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
source_path = ROOT / "data" / "chapters" / "chapters.jsonl"  # 指定上一课生成的章节文件。
SEED = 42  # 固定随机种子，本身没有特殊好坏之分。
with source_path.open(encoding="utf-8") as file:  # 打开章节文件，读取后自动关闭。
    records = [json.loads(line) for line in file]  # 将每行还原成字典，组成记录列表。
assert len({record["id"] for record in records}) == len(records), "输入 ID 重复"  # 先检查输入记录身份唯一。
groups = {}  # 创建字典，按书名分别保存章节。
for record in records:  # 依次处理每条章节记录。
    groups.setdefault(record["book"], []).append(record)  # 书名尚不存在时创建列表，再加入当前章节。
```

### 第二块：按书排序、打乱、切片

```python
rng = random.Random(SEED)  # 创建独立的随机数生成器，不依赖全局随机状态。
splits = {"train": [], "validation": [], "test": []}  # 为三个集合分别准备列表。
for book_name in sorted(groups):  # 固定书籍处理顺序，避免依赖输入记录的排列。
    chapters = sorted(groups[book_name], key=lambda record: (record["chapter_index"], record["id"]))  # 按顺序号和 ID 生成新的有序列表。
    rng.shuffle(chapters)  # 打乱当前列表；字典和章节正文不变。
    count = len(chapters)  # 取得当前书籍的章节数。
    train_end = int(count * 0.8)  # 计算训练集右端点，正数取整相当于向下取整。
    validation_end = train_end + int(count * 0.1)  # 在训练集之后再取约一成章节作为验证集。
    assert 0 < train_end < validation_end < count, "章节太少，无法分成三个非空集合"  # 防止小分组产生空集合。
    splits["train"].extend(chapters[:train_end])  # 将前八成章节逐条加入训练集。
    splits["validation"].extend(chapters[train_end:validation_end])  # 将中间一成章节加入验证集。
    splits["test"].extend(chapters[validation_end:])  # 将剩余章节全部加入测试集，避免取整后遗漏。
    print(book_name, "总章数：", count, "训练：", train_end, "验证：", validation_end - train_end, "测试：", count - validation_end)  # 显示每本书的划分结果。
```

### 第三块：检查互斥与覆盖

```python
ids = {name: {record["id"] for record in rows} for name, rows in splits.items()}  # 为每个集合提取章节 ID 集合。
texts = {name: {record["text"] for record in rows} for name, rows in splits.items()}  # 为每个集合提取完整章节文本集合。
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:  # 检查三种集合配对。
    assert ids[left].isdisjoint(ids[right]), "同一章节进入了多个集合"  # 两个集合不能共享章节 ID。
    assert texts[left].isdisjoint(texts[right]), "不同集合含有完全相同的整章正文"  # 检查不同 ID 下是否存在相同整章文本。
assert ids["train"] | ids["validation"] | ids["test"] == {record["id"] for record in records}, "划分遗漏或多出章节"  # 三个集合合起来必须覆盖全部输入 ID。
assert sum(len(rows) for rows in splits.values()) == len(records), "划分后的总记录数不一致"  # 检查总数没有增加或减少。
print("集合交集、完整章节重复和覆盖检查：通过")  # 所有断言通过后显示。
```

### 第四块：保存划分及内容指纹

```python
output_dir = ROOT / "data" / "splits"  # 指定三个数据集和清单的保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已经存在时继续使用。
file_info = {}  # 保存每个输出文件的记录数、字符数和内容指纹。
for name, rows in splits.items():  # 依次保存训练集、验证集和测试集。
    output_path = output_dir / f"{name}.jsonl"  # 用集合名称生成文件名。
    with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 以 UTF-8 和 LF 覆盖写入当前划分。
        for record in rows:  # 按划分后的固定顺序取出章节。
            file.write(json.dumps(record, ensure_ascii=False) + "\n")  # 每章写成一行 JSON，正文保持不变。
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()  # 根据文件的全部字节计算 SHA-256。
    characters = sum(len(record["text"]) for record in rows)  # 统计当前集合的正文字符总数，不是 JSON 文件字节数。
    file_info[name] = {"count": len(rows), "characters": characters, "sha256": digest}  # 把统计和指纹加入清单。
    print(name, "章节数：", len(rows), "字符数：", characters, "SHA-256：", digest)  # 显示结果，供重复运行时对比。
manifest = {"seed": SEED, "algorithm": "per_book_sorted_shuffle_v1", "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "files": file_info}  # 记录划分配置、输入指纹和输出统计。
manifest_path = output_dir / "manifest.json"  # 指定划分清单位置。
with manifest_path.open("w", encoding="utf-8", newline="\n") as file:  # 打开清单文件并明确换行方式。
    file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")  # 普通 JSON 清单可以用缩进，不受 JSONL 一行一条的要求限制。
print("保存位置：", output_dir)  # 显示本次划分的输出目录。
```

### 原理、参考值与验证

每本书有 K 章，训练章数为 floor(0.8K)，验证为 floor(0.1K)，测试取剩余章数，保证总和仍为 K。当前各书为 96/12/12、96/12/12、96/12/12、80/10/10；总计为 368/46/46。

老师预先按上述算法计算：train 正文 2,456,586 字符，validation 为 317,717，test 为 304,142。各集合字数比例未必恰为 80%/10%/10%，因为章节长度不同。以上是参考值，不代表学习者已完成。

setdefault 在键不存在时保存默认列表，返回该书的列表；extend 把多个章节逐条加入目标列表，append 则会把整个列表当作一个元素。lambda 在这里给 sorted 指定按什么字段排序。sha256 是内容指纹，不衡量数据质量或模型效果。

对总字符数 N、章节数 K、各书章节数 K_b，读取、检查和保存总体为 O(N + K)，排序额外约为 O(sum K_b log K_b)，额外内存约 O(N + K)，哈希需要扫描文件字节。指纹不同只能说明字节不同，可能来自内容、顺序、空格或换行变化。

待学习者连续运行两次，提交代码、章数与字符数、三份输出的两次 SHA-256，并确认四份输出文件存在。老师还将检查重载、无交集、完整覆盖、重复运行以及改变输入记录排列后的稳定性。口述题：验证集和测试集的用途有什么区别？本课完成前不制作训练词表或训练窗口。

### 第五课实操复查：通过

学习者提交的两次运行结果完全一致。老师检查实际脚本，重新运行两次，并独立解析文件验证：

- 训练集 368 章、2,456,586 字符；验证集 46 章、317,717 字符；测试集 46 章、304,142 字符。
- 输出重载后的记录与源章节字典完全相同，四个字段及正文没有修改。
- 460 个章节 ID 唯一，三个集合两两互斥，且合计覆盖全部输入；无相同完整章节正文跨集合。
- manifest 的输入指纹、输出指纹、记录数和字符数均与实际文件相符。
- 连续运行的四个文件字节完全一致。老师在临时目录打乱输入记录的排列后，三个输出数据集字节仍与原输出一致；该场景下输入指纹会变化，因此不要求 manifest 字节一致。
- GitHub 自动检查已加入第五课运行，并比较重新生成的数据划分与提交版本一致。

输出指纹：

```text
train: 0cee762b4832c3a1c37c84c245ca912c51a1c22fa8df4c37e811ec735e390a69
validation: 6c2b342b2964d65d780f8a487285855dfaff4502bf3bb8fbf7bba1896e46e0fe
test: 42133d3767a02aa411b8ce3735c16f8dff6104ff1c8e48ff9abd03654c2c08d2
```

### 口述题复盘：为什么验证与测试分开

学习者理解二者用途不同，并认为验证集判断何时停止、测试集判断真实场景预测准确性，同时提出“不拆分好像也可以”。

补充：验证集虽不参与梯度更新，但通过模型大小、学习率、训练轮次和停止时机等选择间接影响最终模型。若又用同一集合报告最终效果，反复试验可能使模型选择适应该集合的偶然特征，结果偏乐观。独立测试集用来检查方案确定后的表现。当前小说留出章节并不自动代表真实部署场景。

若方案完全预先固定且不根据留出集结果调参，可以只设训练集和一个最终评估集；这与“同时用一个集合调参并作独立最终评估”不同。教学中将进行多次优化，所以保留独立测试集。尚无训练结果，不得据此声称模型效果。

### 新问题：为什么 shuffle

- 不打乱而直接切片，会把每本书前 80% 的章节全部用于训练、中间 10% 用于验证、最后 10% 用于测试，评估可能受故事阶段、人物或写作变化影响。
- 本课目标是评估同批书籍留出的章节，随机分配可减少章节位置造成的系统性偏差；不保证三组主题完全均衡，也不消除章节间相关性。
- 每本书分别 shuffle 后划分，让每个集合都有四本书的数据。随机的是章节列表的排列，不是章内文字的排列。
- 120 章中固定选 96 章训练时，均匀随机排列下每章进入训练集的概率为 96/120 = 0.8；一次固定划分仍可能存在偶然不均衡。
- 数据集划分时的 shuffle 只用于确定集合成员，随后冻结；训练阶段的小批次打乱是另一件事，后续另讲。
- 若目标是根据过去预测未来，应按时间顺序留出；若目标是未见书籍或作者，应按书或作者分组留出，不能认为所有任务都应随机划分章节。

学习者已表示明白验证/测试划分及 shuffle 的补充说明，并要求进入下一课。记录为已确认理解，尚未通过新情境题独立检验。

## 第 6 课：仅从训练集建立字符词表

状态：代码实操与复查通过。学习者已完成 scripts/learn/05_build_vocab.py 及 data/vocab/vocab.json，未知字符不可逆的口述题回答正确。

### 目标与边界

- 为训练集出现的每个字符分配固定整数编号，保留标点、空格和换行。
- 编号只充当索引，不表示大小、语义或相似程度。
- 将 0 保留为特殊编号 <UNK>，代表未见字符。特殊符号是一个词表项，并非将五个字符分别编码。
- 只从训练集收集字符。验证集只用于统计未见字符，不往词表补字；本课不读取测试集。
- 实现 encode、decode，验证训练文字可往返还原。未知字被映射后会丢失具体身份，不能保证任意输入都能还原。
- 输出 data/vocab/vocab.json，固定词表顺序并记录训练文件 SHA-256。

### 第一块：读取训练集并建表

```python
import hashlib  # 导入哈希工具，记录建立词表所用训练文件的指纹。
import json  # 导入 JSON 工具，读取章节并保存词表。
from collections import Counter  # 导入计数器，统计验证集未见字符的出现次数。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
train_path = ROOT / "data" / "splits" / "train.jsonl"  # 指定唯一用于建词表的数据。
validation_path = ROOT / "data" / "splits" / "validation.jsonl"  # 验证集只用于统计未知字符。
UNK_TOKEN = "<UNK>"  # 定义表示未知字符的特殊词表项。
UNK_ID = 0  # 将编号 0 固定保留给未知字符。
with train_path.open(encoding="utf-8") as file:  # 读取训练章节，结束后关闭文件。
    train_records = [json.loads(line) for line in file]  # 将每行 JSON 还原为章节字典。
characters = set()  # 创建集合，收集训练集出现的不同字符。
for record in train_records:  # 逐章查看训练正文。
    characters.update(record["text"])  # 把每个字符加入集合，重复字符只保留一次。
tokens = [UNK_TOKEN] + sorted(characters)  # 未知符号放在首位，其余字符排序后固定编号。
char_to_id = {char: index for index, char in enumerate(tokens)}  # 建立字符或特殊符号到编号的映射。
print("训练集不同字符数：", len(characters))  # 当前应为 5546。
print("含未知符号的词表大小：", len(tokens))  # 当前应为 5547。
```

### 第二块：编码与解码

```python
def encode(text):  # 定义编码函数，把文字转换成编号列表。
    return [char_to_id.get(char, UNK_ID) for char in text]  # 已知字符使用固定编号，未见字符使用 0。

def decode(token_ids):  # 定义解码函数，把编号列表转换成文字表示。
    return "".join(tokens[token_id] for token_id in token_ids)  # 逐个查表连接，0 会显示为 <UNK>。

sample = train_records[0]["text"][:20]  # 从第一条训练记录取前 20 个字符。
encoded = encode(sample)  # 把样例转换成整数列表。
assert decode(encoded) == sample, "训练样例不能还原"  # 已知字符往返转换后应保持不变。
assert all(decode(encode(record["text"])) == record["text"] for record in train_records), "训练文本不能完整还原"  # 逐章检查所有训练文字。
print("样例原文：", repr(sample))  # 显示原始字符及其中的换行。
print("样例编号：", encoded)  # 显示每个字符对应的整数。
print("解码结果：", repr(decode(encoded)))  # 显示查表还原后的文字。
print("未知字符示例：", encode("🧪"), decode(encode("🧪")))  # 当前训练集未收录这个字符，应显示 [0] 和 <UNK>。
```

### 第三块：统计验证集未见字符

```python
unknown_counts = Counter()  # 记录每种未见字符出现了多少次。
validation_characters = 0  # 累计验证集的全部字符数。
with validation_path.open(encoding="utf-8") as file:  # 只读取验证集，不修改词表。
    for line in file:  # 一次读取一条章节记录。
        text = json.loads(line)["text"]  # 从记录中取出完整章节文字。
        validation_characters += len(text)  # 将当前章节长度加入验证集总长度。
        unknown_counts.update(char for char in text if char not in char_to_id)  # 累计所有未收录字符的出现次数。
assert validation_characters > 0, "验证集为空"  # 防止计算未知率时除以零。
unknown_total = sum(unknown_counts.values())  # 计算未见字符出现的总次数，重复出现也计数。
unknown_rate = unknown_total / validation_characters  # 未知率按出现次数计算，不是按种类数计算。
print("验证集字符总数：", validation_characters)  # 当前应为 317717。
print("未见字符种类数：", len(unknown_counts))  # 当前应为 68。
print("未见字符出现次数：", unknown_total)  # 当前应为 102。
print("未知率：", f"{unknown_rate:.4%}")  # 当前约为 0.0321%，注意百分比与小数的区别。
print("最常见的未见字符：", unknown_counts.most_common(10))  # 展示高频未见字符，帮助理解统计结果。
```

### 第四块：保存与重载词表

```python
output_dir = ROOT / "data" / "vocab"  # 指定词表文件目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已存在时继续使用。
output_path = output_dir / "vocab.json"  # 指定词表文件名。
vocab = {"type": "character", "unk_token": UNK_TOKEN, "unk_id": UNK_ID, "tokens": tokens, "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}  # 保存有序词表、特殊编号和训练来源指纹。
with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 以 UTF-8 和 LF 覆盖写入当前词表。
    file.write(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")  # 保存为可阅读的 JSON，并保留 tokens 列表顺序。
with output_path.open(encoding="utf-8") as file:  # 重新打开词表文件。
    loaded_vocab = json.load(file)  # 将整个 JSON 文件还原成字典。
assert loaded_vocab == vocab, "词表保存后发生变化"  # 检查词表顺序、编号配置和来源都未改变。
print("词表路径：", output_path)  # 显示生成位置。
print("训练文本往返及词表重载检查：通过")  # 所有断言均通过后显示。
```

### 原理与参考值

本课仅实现按 Python 字符串元素切分的字符级编码，不做分词或子词训练；组合字形在 Unicode 中可能由多个字符组成，本课不做额外规范化。

训练集中不同字符数为 C，预留一个未知符号后词表大小 V=C+1。当前 C=5546，V=5547，合法编号范围为 0～5546。完整语料此前出现 5675 种字符，训练集没有覆盖所有字符，这是划分后的正常情况，不应拿验证或测试字符补表。

验证未知率 = 未见字符出现次数 / 验证字符总数 = 102/317717，约为 0.0321%；68 是种类数，不能作为该指标的分子。未知率只衡量字符覆盖，不能证明模型生成质量。

对于之后的 RNN，词表大小 V、隐藏层 H，参数量为 VH（字符表示）+H²（循环权重）+H（隐藏偏移）+HV（输出权重）+V（输出偏移），合计 P=2VH+H²+H+V。若 V=5547、H=64，则 P=719723，float32 参数原始字节数为 2878892（约 2.88 MB），不包含训练梯度、优化器状态、激活、词表、Python 对象和文件格式开销，也不是训练峰值内存。

待提交：完整代码、上述统计和示例、词表重载结果，以及“猫、狗两个训练集都没见过的字符如果都变成 0，解码时还能知道原来是哪一个吗”的解释。本课代码与数据尚待实操，不提前记录完成。

### 第六课实操复查：通过

学习者提交完整输出，老师检查并重新运行实际脚本，独立读取词表确认：

- tokens 严格等于 [<UNK>] 加上训练字符集合的排序结果，没有使用验证集或测试集补字。
- 训练字符 5,546 种，词表 5,547 项，token 无重复，未知编号为 0，训练源文件 SHA-256 与实际文件一致。
- 每章训练正文编码解码均可还原，词表重载完全一致，重复运行文件字节一致。
- 验证集总字符 317,717，未见字符 68 种、102 次，未知率约 0.0321%，与参考值一致。
- 老师另外验证两个不同的未见字符 🧪、🧬 均编码成 0，解码只能得到 <UNK><UNK>。
- GitHub 自动检查新增词表重建、编码解码检查和 data/vocab 产物复现检查。

学习者回答“应该不能，因为都被统一编码成了 <UNK>”，正确。多个不同字符映射到同一个编号后，具体身份信息丢失，不能靠查表逆转。

### 新问题：字符编码是否需要先分词

本课已在做字符级 tokenization：每个普通字符作为一个 token，再查表编码；不需要另外先做词级分词。token 不一定是自然语言中的完整词，也可以是字符、子词或其他单位。

例如“我喜欢机器学习。”：

- 字符级示意：我 / 喜 / 欢 / 机 / 器 / 学 / 习 / 。
- 词级示意：我 / 喜欢 / 机器学习 / 。；不同规则可能给出不同切法。
- 子词级则根据具体算法和词表，把文本切成可复用片段，不预设其边界一定与中文词相同。

选择字符级的教学理由：实现容易检查、词表较小、无需引入词级分词器；模型通过上下文学习字符组合，不要求先给完整词分配一个编号。代价是序列通常较长，需要通过多个时间步学习词和短语关系；没收录的字符仍会变为未知符号。

词级会让常见多字词占一个位置，但词表可能更大，新词也可能成为未知项。子词在词表大小与序列长度之间提供另一种取舍，未知输入的覆盖取决于算法及是否有字节回退等机制，不能声称一定没有未知字符。

编码粒度决定词表大小 V 和序列长度 T，两者共同影响模型开销；token 少不代表一定更快。对本课将实现的简单 RNN，单步主要矩阵计算约为 O(H²+HV)，总步骤工作量约为 O(T(H²+HV))，未计批大小和反向传播等常数。字符级词表较小，但同一文本通常需要更多步骤。

后续仍使用当前字符词表，先完整实现训练窗口、模型和训练测评，再把不同 tokenization 作为独立实验比较。下一课计划按章节制作预测下一个字符的输入与目标，窗口不跨章节或数据集。

## 第 7 课：制作下一个字符预测样本

状态：实操复查通过，学习者已完成 scripts/learn/06_make_windows.py。只从训练集生成窗口，不修改已有数据或生成测试样本。

### 核心原理

对“我爱学习。”，输入 x 为“我爱学习”，目标 y 为“爱学习。”。每个位置的答案是输入当前位置的下一个字符，y 不是模型生成的结果，而是从现有语料制作的标准答案。

编号序列为 a，起点 s，长度 L：x=a[s:s+L]，y=a[s+1:s+L+1]，所以必须取得 L+1 个原始编号。两者都是长度 L，且 x[1:]==y[:-1]。

先读取固定词表，不重新编号；校验训练源文件指纹。每次仅在一章内部取样，不先拼接所有章节；当前每个窗口独立处理上下文，后续搭建模型时明确初始隐藏状态。

### 第一块：加载固定词表和训练集

```python
import hashlib  # 导入哈希工具，检查训练文件与建立词表时是否一致。
import json  # 导入 JSON 工具，加载词表和训练章节。
from pathlib import Path  # 导入路径工具，定位数据文件。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
train_path = ROOT / "data" / "splits" / "train.jsonl"  # 本课只读取训练集。
vocab_path = ROOT / "data" / "vocab" / "vocab.json"  # 读取上一课保存的固定词表。
with vocab_path.open(encoding="utf-8") as file:  # 打开词表文件，结束后自动关闭。
    vocab = json.load(file)  # 读取整个词表字典。
tokens = vocab["tokens"]  # 保留已确定的编号顺序，不重新排序。
char_to_id = {char: index for index, char in enumerate(tokens)}  # 由固定列表重建字符到编号的映射。
unk_id = vocab["unk_id"]  # 读取未知字符编号，不在这里重新指定。
assert vocab["train_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest(), "训练文件与词表来源不一致"  # 检查数据版本是否匹配。
with train_path.open(encoding="utf-8") as file:  # 打开训练集文件。
    train_records = [json.loads(line) for line in file]  # 逐行还原章节记录。
```

### 第二块：逐章生成窗口

```python
def iter_windows(records, seq_length, stride):  # 定义生成器，按需提供一个个输入与目标窗口。
    assert seq_length > 0 and stride > 0, "窗口长度和步长必须大于零"  # 防止零长度窗口或无效步长。
    for record in records:  # 每次处理一章，章节之间不拼接。
        token_ids = [char_to_id.get(char, unk_id) for char in record["text"]]  # 将当前章转换成编号列表。
        for start in range(0, len(token_ids) - seq_length, stride):  # 只选择后面还有 L+1 个编号的位置。
            block = token_ids[start:start + seq_length + 1]  # 多取一个编号，用来制作最后一个位置的答案。
            x = block[:-1]  # 去掉最后一个编号，得到 L 个输入。
            y = block[1:]  # 去掉第一个编号，得到向后错开一位的 L 个答案。
            yield {"chapter_id": record["id"], "start": start, "x": x, "y": y}  # 返回当前窗口并暂停，下次请求时继续生成。
```

yield 与 return 不同：yield 提供一个结果并保留执行位置，下一次请求时继续；不会一次把所有窗口存进列表。窗口内容是列表，后续批处理课再转换为 NumPy 数组。

seq_length 是每个窗口预测多少个位置；stride 是相邻窗口起点移动多少个编号。stride=1 会让相邻窗口大量重叠，stride=L 时 x 通常不重叠，但相邻 block 会共享一个用于连接预测的字符。步长大于 L 还可能跳过中间位置，本课用 L=S=32。

### 第三块：短预览和完整计数

```python
preview = next(iter_windows(train_records[:1], seq_length=8, stride=8))  # 先从第一章取一个短窗口，方便逐字检查。
print("样例章节：", preview["chapter_id"], "起点：", preview["start"])  # 记录窗口来自哪一章，起点从 0 开始。
print("输入编号：", preview["x"])  # 显示长度为 8 的输入编号。
print("目标编号：", preview["y"])  # 显示向后错开一位的答案编号。
print("输入文字：", repr("".join(tokens[index] for index in preview["x"])))  # 查表显示输入文字，不删除空格。
print("目标文字：", repr("".join(tokens[index] for index in preview["y"])))  # 查表显示答案文字。

SEQ_LENGTH = 32  # 正式统计时每个窗口预测 32 个位置。
STRIDE = 32  # 相邻窗口起点向后移动 32 个编号。
total_windows = 0  # 从零开始累计窗口数，不把所有窗口保存到列表。
for window in iter_windows(train_records, SEQ_LENGTH, STRIDE):  # 逐个生成训练窗口。
    assert len(window["x"]) == len(window["y"]) == SEQ_LENGTH, "窗口长度不一致"  # 检查输入和答案都足够长。
    assert window["x"][1:] == window["y"][:-1], "输入与答案没有正确错开"  # 检查重叠部分是否符合下一个字符预测。
    total_windows += 1  # 当前窗口检查通过后计数。
expected_windows = sum(max(0, (len(record["text"]) - SEQ_LENGTH - 1) // STRIDE + 1) for record in train_records)  # 用窗口数量公式独立核对。
assert total_windows == expected_windows, "窗口数量与公式不一致"  # 检查循环起点和终点是否正确。
print("训练窗口数：", total_windows)  # 当前应为 76584。
print("预测位置数：", total_windows * SEQ_LENGTH)  # 当前应为 2450688，不等于实际独立字符种类数。
print("窗口长度、错位和数量检查：通过")  # 上述断言均通过后显示。
```

### 边界数学与参考值

一章 N 个编号，最后一个合法起点满足 s+L<N。步长 S 时，若 N>=L+1，则 M=floor((N-L-1)/S)+1，否则 M=0。Python 的 range 右端不包含，因此用 range(0,N-L,S)。N=L 时没有答案所需的第 L+1 个编号，不能生成完整样本。

当前第一条训练记录为 三国演义:010，长度 4227。8 位预览的输入是“第十回　勤王室马”，目标是“十回　勤王室马腾”。输入编号为 [3461,519,857,13,492,2961,1188,5321]，目标为 [519,857,13,492,2961,1188,5321,3871]。注意窗口中全角空格也是一个字符。

L=S=32 时，第一条训练记录有 132 个窗口；整个训练集预计 76,584 个窗口、2,450,688 个预测位置。暂时舍弃每章末尾不足完整窗口的部分，不做补齐。这是长度固定的教学实现，不等于用到了所有训练字符的全部转移；今后可引入变长和掩码。

所有章的编号转换约 O(N)，制作窗口的工作量约 O(ML)，M 为总窗口数；本课 S=L，所以总体随训练字符数大致线性增长。生成器不同时保存所有窗口，但 train_records 已载入内存，当前章的编号列表也占内存，不能把整个程序称为常量内存。

本课没有保存窗口数组，后续按需组成小批次，避免大量重复存储重叠窗口。待提交：完整脚本、8 位预览、窗口总数，以及“输入长为 32，为什么必须取 33 个原始编号”的解释。实现通过后再更新实际结果并上传 GitHub。

### 第七课实操复查：通过

学习者提交输出与参考值一致。老师检查实际脚本并重新运行，独立核对全部 76,584 个窗口，确认 x/y 均为对应 chapter_id、start 下原始编号片段的前 32 位与后 32 位；不存在跨章取样。

另行检查 N=0、31、32、33、64、65 时的窗口数分别为 0、0、0、1、1、2；两章各只有 32 个字符时不能跨章拼出窗口。

学习者回答“如果只取 32 个就无法进行错位”，方向正确但不够精确：32 个编号可以组成 31 对下一字符预测，无法组成长度为 32 的完整输入和答案；第 32 个输入需要第 33 个编号作为目标。

### 新问题：与 Transformer teacher forcing 的关系

错位是输入和监督目标的对齐方式；teacher forcing 是训练时以真实历史 token 作为后续输入，而非使用模型自己预测的 token。当前只是准备数据，后续 RNN 训练直接使用完整真实输入序列时，采用的就是这种训练方式，并非 Transformer 独有。

例如真值为“我爱学习”，模型在“我”后面误预测成“想”，teacher forcing 下下一步仍喂真实的“爱”；自由生成时则可能把采样出的“想”接入前文。

自回归 Transformer 解码器训练也可使用真实序列错位。区别在于它能并行计算多个位置，必须使用因果遮罩阻止位置关注未来 token；仅错位而没有正确遮罩，会泄漏目标。原始 Transformer 编码器和解码器用途不同，不能把所有 Transformer 训练都等同于 teacher forcing。普通单向 RNN 顺序递推，自然限制使用的历史，但也应避免把目标或未来状态另行作为输入。

参考：[PyTorch 官方 seq2seq 教程](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)、[Attention Is All You Need，第 3.1 节](https://arxiv.org/pdf/1706.03762)。

## 第 8 课：把窗口组成 NumPy 小批次

状态：实操复查通过，学习者已在 06_make_windows.py 完成批处理函数与全部批次检查。暂时顺序遍历，不改变已经冻结的数据划分。

### 目标和形状

B 为一个批次的窗口数量，L 为每个窗口长度。输入 X 和目标 Y 均为 (B,L) 的 int64 数组，每行是一条独立窗口，每列为该窗口内的位置。编号不是向量，不要先把编号转换成 float32。批次中可以混合不同章的窗口，但每一行仍保留 chapter_id 与 start，模型之后为各窗口独立初始化隐藏状态。

### 第一块：导入与批次打包函数

在文件顶部增加：

```python
import numpy as np  # 导入 NumPy，把多个窗口转换成规则的整数数组。
```

在现有文件末尾顶格添加：

```python
def pack_batch(windows):  # 把已经收集好的非空窗口列表转换成一个批次。
    x = np.asarray([window["x"] for window in windows], dtype=np.int64)  # 每个输入列表作为一行，组成二维整数数组。
    y = np.asarray([window["y"] for window in windows], dtype=np.int64)  # 按同样行序组成目标数组。
    chapter_ids = [window["chapter_id"] for window in windows]  # 保留每行对应的章节标识。
    starts = [window["start"] for window in windows]  # 保留每行在章节中的起点。
    return {"x": x, "y": y, "chapter_ids": chapter_ids, "starts": starts}  # 返回数组及其来源信息。

def iter_batches(windows, batch_size):  # 将窗口生成器按指定数量分组，逐批提供数据。
    assert batch_size > 0, "批大小必须大于零"  # 防止出现无效批大小。
    buffer = []  # 暂存尚未凑满一批的窗口。
    for window in windows:  # 逐个请求上一课的窗口。
        buffer.append(window)  # 把当前窗口加入临时列表。
        if len(buffer) == batch_size:  # 窗口数达到一批时就打包。
            yield pack_batch(buffer)  # 提供一批数据并暂停，等待下一次请求。
            buffer = []  # 恢复执行后清空缓冲，准备下一批。
    if buffer:  # 遍历结束后，检查是否还有不足一批的窗口。
        yield pack_batch(buffer)  # 保留最后的小批次，不丢弃剩余样本。
```

### 第二块：预览与遍历检查

```python
preview_batch = next(iter_batches(iter_windows(train_records, 8, 8), 4))  # 先取 4 条长度为 8 的窗口，便于观察。
print("预览 X 形状：", preview_batch["x"].shape)  # 应为 (4, 8)，分别代表窗口数和每条长度。
print("预览 Y 形状：", preview_batch["y"].shape)  # 目标形状同样为 (4, 8)。
print("预览数据类型：", preview_batch["x"].dtype)  # 应为 int64，保存整数编号。
print("预览 X 第一行：", preview_batch["x"][0])  # 取第一条输入窗口，不是第一个时间位置的整列。
print("预览 Y 第一行：", preview_batch["y"][0])  # 取与它对应的目标窗口。

BATCH_SIZE = 16  # 正式统计时每批最多取 16 条窗口。
batch_count = 0  # 累计批次数。
sample_count = 0  # 累计真正处理的窗口数。
last_batch_size = 0  # 记录最后一批的实际窗口数。
for batch in iter_batches(iter_windows(train_records, SEQ_LENGTH, STRIDE), BATCH_SIZE):  # 顺序遍历所有训练批次。
    x, y = batch["x"], batch["y"]  # 取出输入与目标数组。
    assert x.ndim == 2 and x.shape == y.shape, "数组形状不正确"  # 两个数组都必须是形状一致的二维数组。
    assert 0 < x.shape[0] <= BATCH_SIZE and x.shape[1] == SEQ_LENGTH, "批次尺寸不正确"  # 最后一批可以不足 16 条，长度仍为 32。
    assert x.dtype == y.dtype == np.int64, "编号必须使用 int64"  # 检查编号没有误用浮点数。
    assert np.array_equal(x[:, 1:], y[:, :-1]), "批次中的目标错位不正确"  # 逐行检查所有时间位置的错位关系。
    assert len(batch["chapter_ids"]) == len(batch["starts"]) == x.shape[0], "来源信息与行数不一致"  # 检查每行都有对应元数据。
    if batch_count == 0:  # 第一个完整批次用于展示编号数组的大小。
        print("首批 X 和 Y 字节数：", x.nbytes + y.nbytes)  # 应为 8192，只计算两个数组的数据区。
    batch_count += 1  # 当前批检查通过后计数。
    sample_count += x.shape[0]  # 按实际行数计数，不假设每批都满。
    last_batch_size = x.shape[0]  # 更新最后一批大小。
assert sample_count == total_windows, "批处理遗漏或重复样本"  # 核对批次处理的总窗口数量。
assert batch_count == (total_windows + BATCH_SIZE - 1) // BATCH_SIZE, "批次数不正确"  # 用整数形式的向上取整公式核对。
print("批次数：", batch_count)  # 应为 4787。
print("处理窗口数：", sample_count)  # 应为 76584。
print("最后一批窗口数：", last_batch_size)  # 应为 8，不是 16。
print("批次形状、错位和覆盖数量检查：通过")  # 所有断言均通过后显示。
```

### 数学与注意事项

76584=4786×16+8，所以保留尾批后共有 ceil(76584/16)=4787 批，最后一批形状为 (8,32)。不能丢弃它后还宣称用到了全部窗口。

int64 每元素 8 字节，一个满批 X 加 Y 的数据区为 2×16×32×8=8192 字节。它不包含数组对象、元数据、生成器缓冲、原始文本、模型参数或训练中间结果，也不是训练峰值内存。批次行之间可以并行计算，但单向 RNN 的时间位置仍需顺序递推，批处理不等于时间维也并行。

待提交：修改后的完整脚本、预览和批次统计，以及“最后一批为什么是 8 行，训练代码为什么不能始终假设有 16 行”的解释。没有训练或生成模型权重，也没有使用验证或测试集。

### 第八课实操复查：通过

学习者提交的统计与参考值一致。老师重新运行实际脚本，并将批次中的每一行与独立窗口迭代器逐项比较，全部 x、y、chapter_id、start 保持一致，76,584 个窗口无丢失、重复或错配。另行验证了空输入、1 个窗口、刚好 16 个窗口、17 个窗口的批次划分和来源顺序。

结果：预览 X/Y 均为 (4,8)、dtype=int64；首个正式批次 X/Y 数据区合计 8192 字节；正式批次 4787 个，总窗口 76584，最后一批 8 行。

### 口述题补充：为什么实际批大小不能固定为 16

学习者表示不知道，以下为补充讲解，不记为独立掌握：76584=4786×16+8，前 4786 批各 16 条，剩余 8 条组成最后一批。batch_size=16 是本课设置的最大窗口数，不保证每批实际都有 16 条。

模型处理最后一批时，输入为 (8,32)，为每个窗口准备的隐藏状态应为 (8,H)。若写死为 (16,H)，将无法与 8 行输入正确对应，可能形状报错或造成错误对齐。损失归一化也应使用实际参与的样本/位置数，而非始终除以 16×32。以后读取 batch_size = x.shape[0]；本课采用保留尾批，不丢弃、不补假样本。

## 第 9 课：模型第一块——字符向量查表

状态：实操复查与两道口述题均通过，学习者已完成 scripts/learn/07_tiny_rnn.py。此时 TinyRNN 类只实现字符表示表和查表，不是完整 RNN，不训练、不生成文字、不保存权重。

### 模型架构与数学

整体路线：整数编号 → 字符向量 → 循环隐藏状态 → 下一个字符分数 → 概率与训练误差。本课只实现第一步映射。

字符编号只充当索引，不能把编号大小理解为语义大小。设置可学习矩阵 E∈R^(V×H)，编号 i 的表示为第 i 行 E[i]。当前 V=5547、H=64，表共有 VH=355008 个参数，float32 原始数据区占 1420032 字节（约 1.42 MB）。本课尚无梯度更新，随机向量没有训练所得的语义。

若输入 X 的形状为 (B,L)，则 Z=E[X] 的形状为 (B,L,H)。同一字符在这个查表层始终得到同一行向量；后续循环层才结合前文构造上下文相关状态。H 是每个字符使用多少个浮点数表示，不是词表大小、窗口长度或批大小。

用 one-hot 行向量乘 E 也能取得同一行，但直接查表不必构造 V 维 one-hot 数组。本层查询和拷贝约 O(BLH)，不能直接套用完整输出层 O(BLHV) 的矩阵乘法开销。

### 第一块：模型类与查表方法

```python
import json  # 导入 JSON 工具，读取固定词表。
from pathlib import Path  # 导入路径工具，找到词表文件。
import numpy as np  # 导入 NumPy，创建参数矩阵并执行查表。

class TinyRNN:  # 先搭建模型类，本课只实现字符向量部分。
    def __init__(self, vocab_size, hidden_size=64, seed=42):  # 接收词表大小、向量维度和初始化种子。
        assert vocab_size > 0 and hidden_size > 0, "模型维度必须为正"  # 防止创建无效参数矩阵。
        self.vocab_size = vocab_size  # 保存词表项数，检查合法编号时会用到。
        self.hidden_size = hidden_size  # 保存向量维度，后续也作为隐藏状态维度。
        rng = np.random.default_rng(seed)  # 创建局部随机数生成器，固定初始参数。
        self.params = {  # 用字典集中保存之后需要训练的参数。
            "embedding": rng.normal(0, 0.1, (vocab_size, hidden_size)).astype(np.float32),  # 生成均值为 0、标准差为 0.1 的随机字符向量表。
        }  # 当前只有这一组参数，后续再加入循环层和输出层。

    def embed(self, token_ids):  # 将二维字符编号转换成三维字符向量。
        token_ids = np.asarray(token_ids)  # 接受列表或数组，但不偷偷把浮点编号转换成整数。
        assert token_ids.ndim == 2, "输入应为二维编号数组"  # 本课规定输入形状为 (B,L)。
        assert np.issubdtype(token_ids.dtype, np.integer), "字符编号必须是整数"  # 防止用浮点数作为索引。
        assert ((token_ids >= 0) & (token_ids < self.vocab_size)).all(), "编号超出词表范围"  # 拒绝负数和越界编号。
        return self.params["embedding"][token_ids]  # 按每个位置的编号取行，得到 (B,L,H)。
```

### 第二块：入口和最小验证

```python
def main():  # 定义演示入口，被其他文件导入时不自动执行。
    root = Path(__file__).resolve().parents[2]  # 找到项目根目录。
    with (root / "data" / "vocab" / "vocab.json").open(encoding="utf-8") as file:  # 打开固定词表。
        vocab = json.load(file)  # 读取词表配置。
    tokens = vocab["tokens"]  # 保留字符编号顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 根据列表位置重建编号映射。
    model = TinyRNN(len(tokens), hidden_size=64, seed=42)  # 创建带有随机表示表的模型雏形。
    sample_texts = ["春天", "春风"]  # 准备两条等长文本，首字相同，方便验证查表结果。
    x = np.asarray([[char_to_id[char] for char in text] for text in sample_texts], dtype=np.int64)  # 得到形状为 (2,2) 的编号数组。
    vectors = model.embed(x)  # 将编号转换成形状为 (2,2,64) 的浮点数组。
    assert vectors.shape == (x.shape[0], x.shape[1], model.hidden_size), "向量形状不正确"  # 从实际输入形状推导输出，不写死批大小。
    assert vectors.dtype == np.float32, "向量应为 float32"  # 编号用整数，查出的参数用浮点数。
    assert np.array_equal(vectors[0, 0], vectors[1, 0]), "同一字符查表结果不一致"  # 两条样例的首字都是春，应取到完全相同的一行。
    assert model.embed(x[:1]).shape == (1, 2, model.hidden_size), "较小批次不能处理"  # 验证减少批次行数后仍能正确查表。
    print("输入编号：", x)  # 显示两条文本对应的整数数组。
    print("参数矩阵形状：", model.params["embedding"].shape)  # 应为 (5547,64)。
    print("字符向量形状：", vectors.shape)  # 应为 (2,2,64)。
    print("字符向量类型：", vectors.dtype)  # 应为 float32。
    print("表示表参数数：", model.params["embedding"].size)  # 应为 355008。
    print("表示表字节数：", model.params["embedding"].nbytes)  # 应为 1420032，仅统计矩阵数据区。
    print("查表、重复字符和动态批大小检查：通过")  # 上述断言均通过后显示。

if __name__ == "__main__":  # 只有直接运行本文件时才启动演示。
    main()  # 调用入口；其他模块导入 TinyRNN 时不会运行演示。
```

main 入口把演示流程与可导入类区分开，为后续复用模型准备。__init__ 在创建实例时运行，self 指当前实例，self.params 是该实例的参数字典；model.embed(x) 内部的 self 对应 model。

待提交：完整脚本、输出，以及“输入 (8,32)，每个字符用 64 个浮点数表示，输出是什么形状；把批大小从 16 改成 8，会改变表示表参数数吗”的解释。通过后再实现循环层，当前没有训练结果。

### 第九课实操与口述复查：通过

学习者输出的参数矩阵为 (5547,64)，样例查表结果为 (2,2,64)、float32，表示表 355008 个参数、1420032 字节。老师检查实际脚本并重新运行，另行验证：

- 批大小为 1、8、16、窗口长为 32 时，查表输出分别为 (1,32,64)、(8,32,64)、(16,32,64)，与直接矩阵索引完全相同。
- 参数矩阵始终为 (5547,64)，参数数不随批大小变化。
- 相同初始化种子产生完全相同的表示表。
- 浮点编号、负编号、越界编号、一维输入均被拒绝。
- 原有重复字符查表一致性和单条输入检查通过；没有修改学习者模型实现。
- GitHub 自动检查新增 07_tiny_rnn.py 演示运行。

学习者回答输入 (8,32) 对应 batch、window_size，输出 (8,32,64) 对应 batch、window_size、向量维度，正确。通常记为 (B,L,H)，其中 H 是 embedding dimension，本课也准备将它作为循环隐藏状态维度。

学习者回答改变批大小不会改变参数量，正确：表示表参数量 P=VH，由词表大小和向量维度决定，批大小只影响一次查表得到的输出数组大小。例如 (8,32,64) 的 float32 输出数据区为 8×32×64×4=65536 字节，不等于模型参数量或完整训练内存。

下一课计划在现有模型类中逐块加入循环权重与隐藏状态，解释 h_t=tanh(e_t+h_(t-1)W_h+b_h)，先验证前向计算、形状与前文依赖，再加入输出层。目前尚无完整 RNN 前向或训练结果。

## 第 10 课：循环层与隐藏状态前向计算

状态：实操复查通过，学习者已在 scripts/learn/07_tiny_rnn.py 完成循环层和检查。口述题尚待回答，不提前记为独立掌握。此课不训练、不保存权重、不输出下一字符概率。

### 核心公式与形状

使用行向量约定：h_t=tanh(e_t+h_(t-1)W_h+b_h)。e_t 为当前字符向量，h_(t-1) 为上一个位置的状态，W_h 为共享循环权重，b_h 为共享偏移。每次前向调用从零状态开始，窗口之间不传递状态。

本实现字符向量维度与隐藏维度均为 H，因此可以直接相加，未额外设置输入变换矩阵；其他 RNN 实现可能显式使用输入投影。

| 数组 | 形状 |
| --- | --- |
| 输入编号 | (B,L) |
| 所有字符向量 | (B,L,H) |
| 某个位置的字符向量 | (B,H) |
| 上一位置及当前状态 | (B,H) |
| 循环权重 | (H,H) |
| 偏移 | (H,) |
| 保存的所有状态 | (B,L,H) |

矩阵乘法 (B,H)@(H,H) 得到 (B,H)，偏移通过广播加到每一行；不是让不同批次行之间交换信息。

### 手算一个更新

设 H=2，e=[0.1,0.2]，h_prev=[0.2,-0.1]，W=[[0.5,0.2],[-0.3,0.4]]，b=[0,0]。

h_prev@W=[0.2×0.5+(-0.1)×(-0.3),0.2×0.2+(-0.1)×0.4]=[0.13,0]。

加上 e 与 b 得 [0.23,0.2]，经过 tanh 后约为 [0.226028,0.197375]。tanh 逐元素进行，提供非线性；数学上结果在 (-1,1)，有限精度计算可能舍入到端点。

### 第一块：替换初始化中的参数字典

仍放在 __init__ 中，保留已有 rng 的创建。用下面整个字典替换原 self.params 字典，不在时间循环里重复创建参数：

```python
        self.params = {  # 集中保存字符表示、循环权重和偏移参数。
            "embedding": rng.normal(0, 0.1, (vocab_size, hidden_size)).astype(np.float32),  # 保留原来的字符向量初始化。
            "recurrent": rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, hidden_size)).astype(np.float32),  # 创建共享循环权重，并按维度缩小初始标准差。
            "bias": np.zeros(hidden_size, dtype=np.float32),  # 隐藏层偏移从零开始，后续训练会调整。
        }  # 每次创建模型只初始化一次，所有时间位置共用这些参数。
```

标准差 1/sqrt(H) 是一种按维度控制初始数值尺度的选择，并不表示唯一正确方案，也不保证训练稳定。本课与之前相同的 seed 及初始化顺序会保留原 embedding 初始化结果。

### 第二块：在类中增加方法

添加到 embed 方法之后、顶格 def main 之前，方法定义缩进 4 个空格：

```python
    def forward_hidden(self, token_ids):  # 按文字顺序计算并保存每个位置的隐藏状态。
        vectors = self.embed(token_ids)  # 将 (B,L) 编号转换成 (B,L,H) 字符向量。
        batch_size, seq_length, _ = vectors.shape  # 从实际输入读取批大小和长度，不固定为 16 或 32。
        assert batch_size > 0 and seq_length > 0, "输入批次和序列不能为空"  # 本课不处理空批次或空序列。
        hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)  # 为每条窗口分别准备全零初始状态。
        states = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)  # 分配数组，保存全部时间位置的状态。
        for t in range(seq_length):  # 按时间位置顺序递推，先算前面的字再算后面的字。
            current = vectors[:, t, :]  # 同时取出各条窗口在当前位置的字符向量。
            hidden = np.tanh(current + hidden @ self.params["recurrent"] + self.params["bias"])  # 结合当前字符和之前状态，得到新状态。
            states[:, t, :] = hidden  # 复制当前状态到对应时间位置，保留完整计算结果。
        return states  # 返回 (B,L,H)，此时还没有输出下一字符分数。
```

hidden 是当前时刻的工作状态，states 保存所有位置的结果。模型参数在前向计算中不更新；状态随位置变化，参数跨位置共享。每次调用重新建零状态，处理的是当前窗口内的前文，不代表记住整个小说。状态是压缩表示，也不是原文的逐字副本。

### 第三块：在 main 末尾增加验证

保留原有示例和断言；在 main 内原有最后一条打印之后添加，整体缩进 4 个空格，不放在 if __name__ 条件之后：

```python
    states = model.forward_hidden(x)  # 对原样例“春天”和“春风”计算状态。
    assert states.shape == (x.shape[0], x.shape[1], model.hidden_size), "隐藏状态形状错误"  # 检查每条输入每个位置都有状态。
    assert states.dtype == np.float32 and np.isfinite(states).all(), "隐藏状态数值异常"  # 检查数据类型及是否出现无穷或无效数。
    assert np.allclose(states[0, 0], states[1, 0]), "后面的字影响了前面的状态"  # 两条输入首字都是春，从零状态开始应相同。
    assert np.allclose(model.forward_hidden(x[:1]), states[:1]), "不同批次行发生干扰"  # 单独计算第一行，应与批量计算对应行一致。
    context_texts = ["春天", "秋天"]  # 两条文本前字不同，但后一个字都是天。
    context_x = np.asarray([[char_to_id[char] for char in text] for text in context_texts], dtype=np.int64)  # 按固定词表编码。
    context_vectors = model.embed(context_x)  # 先查看不包含上下文的字符向量。
    assert np.array_equal(context_vectors[0, 1], context_vectors[1, 1]), "相同字符查表不一致"  # 两个天必须查到相同向量。
    context_states = model.forward_hidden(context_x)  # 再通过循环层加入前文状态。
    assert not np.allclose(context_states[0, 1], context_states[1, 1]), "本例前文变化未影响状态"  # 对当前固定随机参数，两个天的状态应不同。
    print("循环权重形状：", model.params["recurrent"].shape)  # 应为 (64,64)。
    print("隐藏偏移形状：", model.params["bias"].shape)  # 应为 (64,)。
    print("全部隐藏状态形状：", states.shape)  # 应为 (2,2,64)。
    print("同字不同前文的状态不同：", not np.allclose(context_states[0, 1], context_states[1, 1]))  # 当前示例应为 True。
    print("当前总参数数：", sum(param.size for param in model.params.values()))  # 应为 359168，尚不包含输出层。
    print("递推、前文依赖和批次独立性检查：通过")  # 所有断言通过后显示。
```

np.allclose 比较是否在容差内接近，适合浮点计算。示例状态不同只证明前文能影响计算，不代表随机模型已经理解词义；对特殊参数（如循环权重全零）也不保证能观察到差异。

### 参数量、效率与提交

当前参数为 VH+H²+H=5547×64+64×64+64=359168，新增循环层 4160 个参数。长度和批大小不影响参数数；float32 参数数据区为 1436672 字节。

本层时间约 O(BLH²)，保存全部状态约 O(BLH) 空间，字符向量、工作状态等另占空间。B 维同时做矩阵计算，L 维需逐步递推。训练时还需要保存和反传其他中间结果，本课不是训练峰值内存测量。

待提交：完整代码、末尾递推检查输出，并回答：输入为 (8,32) 时 hidden 和 states 分别是什么形状？“春天”和“秋天”里的“天”查表向量相同，为什么隐藏状态可以不同？本课尚未加入输出层、损失函数或训练更新。

### 第十课实操复查：通过

学习者提交了全部运行输出，与参考值一致：循环权重 (64,64)、偏移 (64,)、样例全部状态 (2,2,64)、同字不同前文的状态不同为 True、总参数 359168。

老师检查实际脚本并重新运行，另外独立按公式逐步计算，验证：

- 输入形状 (1,1)、(8,32)、(16,32) 的返回结果分别为 (1,1,64)、(8,32,64)、(16,32,64)，dtype 为 float32，与独立递推结果一致。
- 修改最后一个输入字符，不影响之前位置的状态，当前计算没有使用未来字符。
- 单独计算一行与批量计算对应行相同，没有批次行之间的状态干扰。
- 重复前向调用结果一致，状态从零重新开始；全部参数在前向计算后保持不变，没有误把状态更新当成参数训练。
- 老师仅更新类旁已过期的说明注释，未代改学习者计算实现。
- GitHub 自动检查运行现有演示，覆盖新增的递推、前文依赖和独立性断言。

学习者已回答形状为 hidden=(32,64)，states 猜测为 (64,64)，需要纠正：给定输入 (8,32)、H=64，当前 hidden 为 (8,64)，全部 states 为 (8,32,64)。states[0] 才是单条窗口的全部历史状态，形状为 (32,64)；(64,64) 是共享循环权重矩阵的形状，不是状态历史。每一步仅更新 8 条窗口当前的状态，32 个位置由循环逐步处理，states 将 32 份 (8,64) 的快照沿时间轴保存。

第二题抓住了前文不同会导致数值计算不同，但将前向、权重变化与反向传播混在了一起。应修正：两个“天”使用同一个字符向量、同一个循环权重和同一个偏移，上一状态可能分别来自“春”和“秋”，因此当前状态可以不同。这个差异仅依靠固定参数的前向计算就能出现，不依赖反向传播或参数更新。tanh 的输出值不同，不表示为两个语境切换了不同的权重。本课没有执行训练或计算梯度。

区分三个过程：前向计算状态和输出；反向计算损失相对参数的梯度；优化器根据梯度更新参数。目前只完成第一种过程的一部分。记录为已补充讲解，不能将原回答判为全部正确，也不提前记录为独立掌握。

下一课计划加入输出权重和输出偏移，将 (B,L,H) 状态投影为 (B,L,V) 的候选字符分数。分数并非概率，概率转换及训练误差后续逐步实现。

## 第 11 课：输出层与候选字符分数

状态：代码实操复查通过，学习者已完成输出层。口述形状题需纠正，分数与概率题已补充讲解。本课未训练、未保存权重；下一课再实现概率和误差。

### 输出层原理与手算

每个位置的状态 h 是 H 维向量，要对 V 个候选字符分别评分。引入 W_o∈R^(H×V) 与 b_o∈R^V，计算 z=hW_o+b_o。第 j 个字符的分数为 z_j=sum_i h_i W_o[i,j]+b_o[j]。所有位置共享这套输出权重，且它与字符表示表是两组独立参数。

形状为 (B,L,H)@(H,V)→(B,L,V)，NumPy 的 @ 对最后两个维度执行矩阵乘法，保留前面的批次维度。输出每个位置有 V 个分数，不是直接得到一个字符编号。

手算 H=2、V=3：h=[0.2,-0.1]，W_o=[[1,0,-1],[0.5,2,1]]，b_o=[0,0.1,0]，结果 z=[0.15,-0.1,-0.3]。分数可以为负，也不要求总和为 1；不能直接当作概率。

### 第一块：增加输出参数

在 __init__ 的 self.params 字典内，bias 行之后、右花括号之前增加两行：

```python
            "output": rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, vocab_size)).astype(np.float32),  # 为每个候选字符准备一列输出权重。
            "output_bias": np.zeros(vocab_size, dtype=np.float32),  # 给每个候选字符准备一个偏移，初始为零。
```

保留之前的参数及 rng，不重新初始化它们。把类旁“尚未加入输出层”注释改为已经包含字符表示、循环层和输出层。原 main 中总参数打印的旧预期注释由 359168 更新为 719723。

### 第二块：增加完整前向方法

在类内 forward_hidden 方法之后、顶格 def main 之前，缩进 4 个空格添加：

```python
    def forward(self, token_ids):  # 从字符编号计算每个位置的候选字符原始分数。
        states = self.forward_hidden(token_ids)  # 先得到 (B,L,H) 的上下文状态。
        logits = states @ self.params["output"] + self.params["output_bias"]  # 使用共享输出层得到 (B,L,V) 分数。
        return logits  # 返回原始分数，暂不做 softmax，也不计算损失。
```

### 第三块：在 main 末尾验证

在原 main 的最后一条打印之后，保持 4 个空格缩进添加：

```python
    logits = model.forward(x)  # 对原样例计算所有位置的下一字符分数。
    assert logits.shape == (x.shape[0], x.shape[1], len(tokens)), "输出分数形状错误"  # 每条窗口每个位置都要对整个词表评分。
    assert logits.dtype == np.float32 and np.isfinite(logits).all(), "输出分数数值异常"  # 检查类型以及数值是否有限。
    expected_scores = states[0, 1] @ model.params["output"] + model.params["output_bias"]  # 单独手工计算第一条窗口第二个位置的输出。
    assert np.allclose(logits[0, 1], expected_scores, rtol=1e-5, atol=1e-6), "输出层计算不符合公式"  # 允许 float32 不同计算顺序造成的微小误差。
    assert np.allclose(logits[0, 0], logits[1, 0], rtol=1e-5, atol=1e-6), "相同前文的分数不一致"  # 两条原样例首字都是春，应给出接近的下一字分数。
    assert np.allclose(model.forward(x[:1]), logits[:1], rtol=1e-5, atol=1e-6), "单条与批量前向不一致"  # 检查实际批大小变化不影响对应结果。
    print("输出权重形状：", model.params["output"].shape)  # 应为 (64,5547)。
    print("输出偏移形状：", model.params["output_bias"].shape)  # 应为 (5547,)。
    print("全部分数形状：", logits.shape)  # 应为 (2,2,5547)。
    print("首条末位置的前五项分数：", logits[0, -1, :5])  # 仅展示词表前五项的分数，不是概率或前五名。
    print("含输出层的总参数数：", sum(param.size for param in model.params.values()))  # 应为 719723。
    print("输出层形状、公式和批次独立性检查：通过")  # 所有断言通过后显示。
```

states 已由前面的 main 代码计算，不删除旧代码。本课新增参数放在原参数之后，先前随机初始化顺序不变，因此旧 states 与 model.forward 内重新计算的状态一致。

本课显式给 allclose 设置 rtol=1e-5、atol=1e-6。float32 的矩阵乘法与单向量乘法可能采用不同累加顺序，结果可能存在很小的舍入差异；接近零的结果尤其需要合适的绝对误差容限。老师预先执行上述代码时已检查这一点，不能把舍入差异直接判成公式错误。

### 分数位置含义与效率

对于输入“春天”，logits[0,0] 是读到“春”后对下一个字符的评分，训练目标可能是“天”；logits[0,1] 是读到“春天”后对再下一个字符的评分，而非对当前“天”分类。窗口外的下一个真实字符由此前制作的 y 提供。

输出层新增 HV+V=64×5547+5547=360555 个参数，全模型 359168+360555=719723，float32 参数数据区为 2878892 字节。参数仍不依赖 B 或 L。

输出计算约 O(BLHV)，保存全部 logits 需 O(BLV) 空间，可能比状态大很多。B=16、L=32、V=5547 时，仅 logits 的 float32 数据区为 11360256 字节（约 11.36 MB），并非模型文件大小或训练峰值内存。本课仍为随机未训练模型，不根据随机分数高低宣称有生成能力。

待提交：完整脚本、末尾输出，以及“输入 (8,32)、V=5547 时 logits 什么形状；负分数为什么不是负概率”的解释。实操通过后再记录真实结果并上传 GitHub。

### 第十一课实操复查：通过

学习者输出与参考一致：输出权重 (64,5547)、偏移 (5547,)、样例分数 (2,2,5547)，总参数 719723。老师重新运行实际脚本，并验证：

- 输入形状 (1,1)、(8,32)、(16,4) 均得到相应 (B,L,5547) 的有限 float32 分数。
- 单个位置逐项按公式计算，与批量输出在 float32 容差内一致。
- 单独一行与批量对应行一致，修改最后一个字符不影响之前位置分数。
- 前向计算前后全部参数保持不变，没有把前向运算当作训练更新。
- 老师仅修正两处过期注释和一处行尾空白，未代改学习者计算逻辑。
- GitHub 自动检查已覆盖输出层演示及断言。

### 口述题复盘：矩阵乘法的求和维度

学习者认为 (8,32,64) 与 (64,5547) 相乘得到 (8,32,64,5547)，错误。矩阵乘法不是拼接维度；两个相匹配的 64 是被求和的维度，不保留为额外轴。

对某一条窗口 b、某个位置 t、某个候选字符 j，z[b,t,j]=sum_i(states[b,t,i]×W_o[i,j])+b_o[j]，i 从 0 到 63。每 64 个对应乘积加成一个分数；共 5547 列，得到 5547 个分数。8 条窗口各有 32 个位置，因此最终形状为 (8,32,5547)。

循环层实际每次执行 (8,64)@(64,64)，得到 (8,64)，依次重复 32 个位置；不能忽略时间依赖而把所有循环状态同时独立计算。输出层在状态已全部得到后，才可执行 (8,32,64)@(64,5547) 的批量投影。NumPy 中 @ 是矩阵乘法，* 是逐元素乘法，二者规则不同。

### 口述题复盘：负分数不等于负概率

学习者表示不知道。补充：logits 是未归一化的评分，可为任意实数，没有概率的非负和总和为 1 的要求。转换为概率可以使用 softmax：p_j=exp(z_j)/sum_k exp(z_k)，分母遍历全部候选，而非仅预览的前五项。

此前手算分数 [0.15,-0.1,-0.3] 对应 softmax 约为 [0.413834,0.322294,0.263872]；负分数也对应非负概率。分数的绝对正负不能单独解释为概率高低，需要比较同一位置的其他候选。概率转换不改变 (B,L,V) 形状，只改变各位置候选数值的意义。

当前只记录代码实操通过，形状与概率概念已详细讲解，尚未通过复答确认独立掌握。下一课计划实现沿最后一维的稳定 softmax，解释减去最大分数为什么不改变结果，再逐步实现负对数似然；暂不加入反向传播。

### 第十一课形状复习题

老师问 (3,5,16) 的状态与 (16,100) 的输出权重相乘后的形状，学习者回答 (3,5,100)，正确。记录为该复习题通过，并据此进入下一课，不将尚未复答的其他概念一并记为全部掌握。

## 第 12 课：数值稳定的 softmax

状态：代码实操复查通过，学习者已实现独立 softmax 和全部演示检查。口述题已答对归一化轴，但求和形状及平移不变的原因仍需补充。本课不计算训练损失、不计算梯度、不更新参数。

### 数学与归一化维度

同一位置有 V 个分数 z，定义 p_j=exp(z_j)/sum_k exp(z_k)。对模型输出 (B,L,V)，只沿最后一维 V 计算，每条窗口的每个位置分别归一化，形状仍为 (B,L,V)，沿最后一维求和后形状为 (B,L)，每项约为 1。

为避免对大正数直接求指数，先令 c=max_j z_j，再计算 exp(z_j-c)/sum_k exp(z_k-c)。分子分母同时有公共因子 exp(-c)，在数学上抵消，因此不改变 softmax。每个位置单独取自己的最大值，不在整个批次中只取一个最大值。

减去最大值后，各输入不大于 0，至少一个为 0，指数最大为 1，分母至少为 1，不需要额外给分母加 epsilon。很小的概率仍可能因浮点下溢变成 0，所以代码检查非负，而非要求每项严格大于 0。下一课的对数损失将使用稳定的对数计算，而不假设所有 softmax 概率都能无损保存。

### 第一块：独立函数

在 imports 之后、class TinyRNN 之前顶格添加；该函数没有可学习参数，不需要放进模型类，也不改变 forward 返回 logits 的约定。

```python
def softmax(logits):  # 将最后一维的候选分数转换为概率，保留其余维度。
    logits = np.asarray(logits)  # 接受浮点列表或数组，已有浮点数组不强制改变精度。
    assert logits.ndim >= 1 and logits.shape[-1] > 0, "候选维度不能为空"  # 至少要有一个可归一化的维度。
    assert np.issubdtype(logits.dtype, np.floating), "分数应为浮点数"  # 不在这里把整数分数静默转换成浮点数。
    assert np.isfinite(logits).all(), "分数包含无穷或无效数"  # 本课只处理有限分数。
    shifted = logits - logits.max(axis=-1, keepdims=True)  # 每个位置减去自己的最大分数，并保留最后一维用于广播。
    exp_scores = np.exp(shifted)  # 对平移后的分数求指数，避免大正分数引起指数溢出。
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # 每个候选除以同一位置所有候选的指数和。
```

axis=-1 表示最后一维。keepdims=True 让最大值及指数和的形状为 (B,L,1)，可以广播到 (B,L,V)。代码用减法创建新数组，不原地改写输入 logits。

### 第二块：在 main 末尾验证真实输出

在 main 原最后一条打印之后添加，保持 4 个空格缩进：

```python
    logits_before = logits.copy()  # 保存分数副本，用于确认转换概率没有改写原分数。
    probabilities = softmax(logits)  # 将 (B,L,V) 原始分数转换成同形状概率。
    probability_sums = probabilities.sum(axis=-1)  # 分别计算每个位置全部候选的概率和，得到 (B,L)。
    assert probabilities.shape == logits.shape, "概率形状改变了"  # softmax 不增加或删除数据维度。
    assert probabilities.dtype == logits.dtype and np.isfinite(probabilities).all(), "概率类型或数值异常"  # 当前 float32 分数应得到有限 float32 概率。
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), "概率超出范围"  # 浮点计算允许极小概率下溢成零。
    assert np.allclose(probability_sums, 1.0, rtol=1e-5, atol=1e-6), "每个位置的概率和不为一"  # 求和检查针对所有候选，不只看预览的五项。
    assert np.array_equal(logits, logits_before), "softmax 修改了输入分数"  # 保留原始 logits 供后续稳定损失计算使用。
    print("概率形状：", probabilities.shape)  # 当前应为 (2,2,5547)。
    print("各位置概率和：", probability_sums)  # 当前为 (2,2)，每项在浮点容差内接近 1。
    print("首条末位置的前五项概率：", probabilities[0, -1, :5])  # 只是词表前五项，不要求这五项加起来等于 1。
    print("概率范围、归一化和输入保留检查：通过")  # 所有断言通过后显示。
```

### 第三块：大分数与平移不变性

继续在 main 内添加，仍缩进 4 个空格：

```python
    large_scores = np.array([1000.0, 1001.0, 999.0], dtype=np.float64)  # 构造直接求指数可能溢出的分数。
    large_probabilities = softmax(large_scores)  # 稳定实现先减去最大值，实际对 [-1,0,-2] 求指数。
    shifted_probabilities = softmax(large_scores - 1000.0)  # 所有候选同时减去同一个常数，数学结果应不变。
    assert np.isfinite(large_probabilities).all(), "大分数转换后出现异常"  # 检查没有因指数溢出出现无穷或无效概率。
    assert np.allclose(large_probabilities, shifted_probabilities, rtol=1e-8, atol=1e-10), "统一平移改变了概率"  # 用 float64 样例验证平移不变性。
    assert np.allclose(large_probabilities.sum(), 1.0), "样例概率和错误"  # 检查这个一维样例也被正确归一化。
    print("大分数转换后的概率：", large_probabilities)  # 约为 [0.24472847,0.66524096,0.09003057]。
    print("大分数与统一平移检查：通过")  # 上述断言均通过后显示。
```

### 概率含义、效率与待提交

最高分候选也有最高 softmax 概率，转换只表达模型当前分配的相对概率，不能让随机参数获得知识或提高准确率。当前 V=5547，许多概率约为万分之几很正常；要按全部候选归一化，不能把预览五项当成完整候选集。

softmax 计算约 O(BLV)，额外数组也约 O(BLV)，不新增任何可训练参数，全模型仍为 719723 个参数。实际训练时可由 logits 直接计算稳定的 log-softmax/交叉熵，不必总是保存一份完整概率用于损失。

待提交：完整脚本、概率检查输出、大分数样例，以及“(8,32,5547) 的概率应沿哪个维度归一化，求和后的形状是什么；为什么同一位置的所有分数减去同一个数不改变概率”的解释。实操通过后更新结果和 CI 并完整上传 GitHub。

### 第十二课实操复查：通过

学习者输出与参考一致：概率形状 (2,2,5547)，各位置概率和约为 1，float32 下出现 1.0000001 属于容差内误差，大分数样例得到 [0.24472847,0.66524096,0.09003057]。

老师重新运行实际脚本，并额外验证：

- 对 (8,32,7) 分数最后一维归一化，输出形状不变；默认求和后为 (8,32)，keepdims=True 后为 (8,32,1)。
- 给每个位置分别加上该位置自己的统一常数，不改变该位置概率；不是只验证整个批次共用一个常数。
- 分数输入在 softmax 后保持不变，等分数给出均匀概率，单候选概率为 1，大分数保持有限。
- 整数输入、NaN、无穷以及空候选维度会被拒绝。
- 老师未改动学习者的 softmax 计算实现，GitHub 自动检查覆盖其演示及归一化断言。

### 口述题补充

学习者正确选择第三维 5547 为归一化轴，但尚未给出求和后形状。解释：原数组 (8,32,5547) 表示 8 条窗口、每条 32 个位置、每位置 5547 项概率；把每组 5547 项加为一个总和，默认得到 (8,32)。使用 keepdims=True 则把最后一维保留为长度 1，得到 (8,32,1)。softmax 本身输出仍为 (8,32,5547)，这和额外求和后的形状不同。

学习者第二题回答“因为平移不变性”，说出了性质名称，但尚未解释原因。需补充：exp(z_j-c)=exp(z_j)×exp(-c)，同一位置所有候选都减同一个 c，因此分子、分母都乘上同一个正数 exp(-c)，相除时抵消。不是所有数值变换都会保持概率，只因该变换在指数后产生共同乘数。

数值例子：[1000,1001,999] 减去最大值 1001 得 [-1,0,-2]；数学上不改变概率，同时避免直接对大正数求指数。某个候选独自减不同数则一般会改变概率。选择最大值使所有平移分数不大于 0，且至少一个为 0，分母因此至少含一项 exp(0)=1。

当前记录为代码实操通过、归一化轴回答正确、形状与不变性原因已补充讲解，不提前记为全部独立掌握。下一课计划用真实目标字符的负对数概率衡量预测误差，使用稳定 log-softmax，保持当前 softmax 不被误用于先取 log 的不稳定计算。

## 第 13 课：稳定的对数概率与交叉熵损失

状态：代码实操复查通过，学习者已实现稳定 log_softmax 与 cross_entropy；交叉熵含义、平均方式和输出形状需补充讲解。本课没有反向传播、参数更新或模型效果评估。

### 从正确答案概率到损失

某一位置的正确标签为 y，损失为 -ln(p_y)，ln 为自然对数，NumPy 用 np.log。若正确概率为 0.8、0.5、0.1，损失约为 0.223144、0.693147、2.302585，正确答案分配的概率越低，损失越大。

单一确定标签时，交叉熵等于该标签的负对数概率，也称负对数似然。不是对全部候选分别取负对数再平均。先在 V 个候选中取正确答案的那一项，再对 B×L 个有效预测位置取平均。本课没有 padding，所以全部位置都有效；以后引入 padding 需要按掩码排除。

模型分数形状 (B,L,V)，目标编号形状 (B,L)，逐位置损失形状 (B,L)，最终平均损失为一个标量。归一化使用实际 B×L，因此尾批也能正确计算。

### 为什么不直接 log(softmax)

对于分数 [1000,0]，真实目标为第二项，其概率极小，浮点数可能将其存成 0，之后 log(0) 得到负无穷。采用 log_softmax：先 shifted=z-max(z)，再计算 log_probs=shifted-log(sum(exp(shifted)))。该例第二项对数概率约 -1000，损失为 1000，仍可表示；不靠加 epsilon 截断真实大损失。

### 第一块：在类外增加稳定对数概率

在 softmax 函数之后、class TinyRNN 之前顶格添加，保持原函数及 forward 返回 logits 的约定：

```python
def log_softmax(logits):  # 直接计算稳定的对数概率，不先保存可能下溢的概率。
    logits = np.asarray(logits)  # 接受浮点列表或数组，保留已有浮点精度。
    assert logits.ndim >= 1 and logits.shape[-1] > 0, "候选维度不能为空"  # 至少要有一项候选。
    assert np.issubdtype(logits.dtype, np.floating), "分数应为浮点数"  # 不把整数编号当成分数。
    assert np.isfinite(logits).all(), "分数包含无穷或无效数"  # 本课要求输入为有限数值。
    shifted = logits - logits.max(axis=-1, keepdims=True)  # 每个位置单独减去最大分数。
    log_denominator = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))  # 计算平移后指数和的自然对数。
    return shifted - log_denominator  # 利用 log(a/b)=log(a)-log(b)，得到同形状对数概率。
```

### 第二块：按真实标签取值并求平均

紧接在 log_softmax 后面顶格添加：

```python
def cross_entropy(logits, targets):  # 输入候选分数和正确编号，返回平均损失与逐位置损失。
    log_probs = log_softmax(logits)  # 得到 (B,L,V) 的对数概率。
    targets = np.asarray(targets)  # 保留标签整数类型，不静默转换浮点标签。
    assert log_probs.ndim == 3 and targets.shape == log_probs.shape[:2], "分数与标签形状不匹配"  # 标签每个位置只有一个编号。
    assert targets.size > 0, "没有可计算损失的位置"  # 防止对空数组取平均。
    assert np.issubdtype(targets.dtype, np.integer), "目标编号必须是整数"  # 标签用于索引候选，不能使用浮点数。
    assert ((targets >= 0) & (targets < log_probs.shape[-1])).all(), "目标编号越界"  # 每个标签必须在词表编号范围内。
    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])  # 合并批次和时间轴，得到 (B×L,V)。
    flat_targets = targets.reshape(-1)  # 按相同顺序展平标签，得到 (B×L,)。
    rows = np.arange(flat_targets.size)  # 为每个预测位置生成一个行号。
    correct_log_probs = flat_log_probs[rows, flat_targets]  # 每行只取正确标签那一列，不取整个候选向量。
    position_losses = -correct_log_probs.reshape(targets.shape)  # 取负号并恢复为 (B,L) 的逐位置损失。
    return float(position_losses.mean()), position_losses  # 按实际位置数求平均，并保留逐位置结果供检查。
```

reshape(-1,V) 的 -1 表示根据元素总数推算这一维长度，不表示复制、打乱或丢掉数据。形状 (2,2,V) 变为 (4,V)，标签 (2,2) 对应变为 (4,)。rows=[0,1,2,3] 与 flat_targets 配对，选出四个正确候选的数值。没有构造整份 one-hot 标签数组。

### 第三块：main 内用完整短句制作答案

继续在 main 原末尾添加，整体缩进 4 个空格。以下是专门手写的演示文本，不声称来自语料，不用于测评质量：

```python
    toy_texts = ["春天来", "春风吹"]  # 指定完整短句，确保最后一个输入也有真实下一字作为答案。
    toy_x = np.asarray([[char_to_id[char] for char in text[:-1]] for text in toy_texts], dtype=np.int64)  # 得到“春天”“春风”两个输入。
    toy_y = np.asarray([[char_to_id[char] for char in text[1:]] for text in toy_texts], dtype=np.int64)  # 得到错位后的“天来”“风吹”两个目标。
    toy_logits = model.forward(toy_x)  # 用当前随机模型计算每个位置的候选分数。
    loss, position_losses = cross_entropy(toy_logits, toy_y)  # 选取正确目标的对数概率，计算逐位置和平均损失。
    assert position_losses.shape == toy_y.shape, "逐位置损失形状错误"  # 本例应为 (2,2)。
    assert np.isfinite(position_losses).all() and (position_losses >= 0).all(), "损失数值异常"  # 交叉熵应有限且非负。
    assert np.isclose(loss, position_losses.sum() / toy_y.size), "损失平均方式错误"  # 核对按实际四个位置求平均。
    print("目标编号形状：", toy_y.shape)  # 应为 (2,2)，不是 (2,2,5547)。
    print("逐位置损失：", position_losses)  # 每个位置一个数，越小表示正确目标的概率越高。
    print("平均交叉熵：", loss)  # 随机模型在这个演示上的误差，不是训练后效果。
```

### 第四块：两个可手算验证例

继续在 main 内添加，保持 4 个空格缩进：

```python
    known_logits = np.log(np.array([[[0.2, 0.5, 0.3]]], dtype=np.float64))  # 构造 softmax 后恰为指定概率的分数，形状为 (1,1,3)。
    known_targets = np.array([[1]], dtype=np.int64)  # 正确答案为编号 1，其概率是 0.5。
    known_loss, _ = cross_entropy(known_logits, known_targets)  # 对一个位置计算交叉熵。
    assert np.isclose(known_loss, -np.log(0.5)), "手算损失不一致"  # 应约为 0.693147。
    extreme_logits = np.array([[[1000.0, 0.0]]], dtype=np.float64)  # 第二项分数远低于第一项，普通概率可能下溢。
    extreme_targets = np.array([[1]], dtype=np.int64)  # 指定低概率的第二项为正确答案。
    extreme_loss, _ = cross_entropy(extreme_logits, extreme_targets)  # 直接由稳定对数概率计算大损失。
    assert np.isfinite(extreme_loss) and np.isclose(extreme_loss, 1000.0), "极端分数损失不稳定"  # 不应得到无穷，也不应被 epsilon 截断成较小值。
    print("概率为 0.5 的正确答案损失：", known_loss)  # 应约为 0.693147。
    print("极端低概率目标的损失：", extreme_loss)  # 应为 1000。
    print("目标索引、平均方式和稳定损失检查：通过")  # 所有断言通过后显示。
```

### 基线、效率与待提交

若 V 项分数完全相同，则每项概率为 1/V，每个正确目标的损失为 ln(V)。V=5547 时约为 8.621012；这是严格均匀分布的数学基线，不要求任意随机模型的样例损失恰好等于该值。不得把当前少量手写例子当成完整训练或验证损失。

稳定 log_softmax 扫描所有候选，时间和主要中间数组约 O(BLV)，目标索引额外约 O(BL)，不增加模型参数。计算一个误差值不等于模型已经学习；后续还需要梯度和更新。

待提交：完整代码、逐位置损失和两组验证输出。口述题：正确答案概率 0.8 和 0.2，哪个损失更大，为什么？当 logits=(8,32,5547)、targets=(8,32) 时，逐位置损失与平均损失分别是什么形状？实操通过后再更新结果、CI 和 GitHub。

### 第十三课实操复查：通过

学习者提交结果与参考值一致：逐位置损失约 [[8.769016,8.818549],[8.527371,8.654839]]，平均约 8.69244385；已知正确概率 0.5 的损失约 0.693147；极端低概率目标的稳定损失为 1000。

老师重新运行实际脚本，另行手工按每个位置的目标编号计算对照，确认逐位置损失、标量平均均正确；另外验证统一分数平移、重复整批数据时平均损失不变、(8,32) 逐位置结果、均匀概率基线和非法标签拒绝。老师仅清理空白格式，不改学习者计算实现。

学习者正确比较 -ln(0.8)<-ln(0.2)，并提出为什么平均、为什么交叉熵、交叉熵是什么，以及不理解损失形状。以下记录补充讲解，不提前视作独立掌握。

### 交叉熵的含义

一个位置的标签分布记为 q，预测分布记为 p，交叉熵 H(q,p)=-sum_j q_j ln p_j。它用目标分布作为权重，计算模型预测概率的平均负对数代价，不是严格的几何距离。

本任务一个训练位置只有一个观察到的正确编号。例如候选为天/风/雨，目标为天，标签写成 q=[1,0,0]，模型 p=[0.8,0.1,0.1]，交叉熵为 -(1×ln0.8+0×ln0.1+0×ln0.1)=-ln0.8。本实现直接用整数标签取正确项，无需构造 one-hot。one-hot 表达当前样本标签，不表示语言里其他续写永远不合法。

准确率只判断最高概率项是否正确；正确答案概率为 0.51 和 0.99 时可能都算预测正确，但交叉熵能区分其概率分配，并为后续梯度训练提供连续的数值目标。交叉熵并非唯一可选损失；此处适合预测离散下一字符概率，也等价于最小化负对数似然、最大化语料中已观察续写的概率。对序列，是给定真实前文条件下各位置概率的乘积，取负对数后变成可相加的各位置损失。

### 为什么求平均，是否必须

求和与求平均都可用。固定预测位置数 N 时，平均只是总损失除以常数 N，不改变最优参数的位置，但会改变梯度的数值尺度。使用平均可减少损失数值和梯度尺度随批内位置数增加而增加的影响，便于比较不同大小批次，也能处理尾批。

例如损失列表 [2,2] 总和为 4，[2,2,2,2] 总和为 8；后者不表示每个位置预测更差，二者平均都为 2。本课定义 loss=sum(position_losses)/(实际 B×L)，并非除以 V，也不是始终除以预设最大批大小乘长度。

以后报告整个数据集的平均损失，应累计所有有效位置的损失总和，再除以总位置数。不能将 16 行满批与 8 行尾批的批平均不加权再平均。当前单批平均是损失定义，训练采样和优化效果仍需通过实验判断。

### 损失形状逐步解释

logits=(8,32,5547) 表示 8 条窗口、各 32 个位置、每位置 5547 个候选分数。targets=(8,32) 为每位置指定一个正确候选。选取正确候选的对数概率后，只剩 8×32=256 个数，恢复形状为 (8,32)，取负后仍为 (8,32)。然后对这 256 个数求平均，得到一个数。

因此 position_losses.shape=(8,32)，loss 是一个 Python float，不是还带有 (8,32) 形状的数组；虽然数学上标量常记为空形状 ()，本函数返回的 Python float 本身没有 .shape 属性。代码中 reshape(-1,V) 和标签展平仅把位置排列成行，不改变各位置对应关系。

本课暂停增加代码，先让学习者理解目标概率→逐位置损失→标量平均的路径。待复答：若两条窗口每条有三个预测位置，一共有几个逐位置损失、它们排成什么形状，平均后有几个数？

### 第十三课损失形状复答

学习者回答两个窗口、三个预测位置共有 6 个逐位置损失，形状 (2,3)，平均后一个数，正确。记录为该复习题通过，不将其他尚未独立检验的损失概念一并视为完全掌握。

## 第 14 课：数值梯度与第一次参数更新

状态：实操复查通过，学习者已完成 scripts/learn/08_gradient_demo.py。此课是独立的单位置、单参数演示，不修改已有 RNN 参数，不使用语料训练，不保存权重。更新方向原理需补充讲解。

### 为什么先做一个参数

设一个位置有三个候选“天、风、雨”，正确答案是第 0 项“天”。原始分数为 [w,0,0]，只允许调节 w；可把它理解为输出层某一个候选的可调偏移，其余分数固定。开始 w=0，三个候选概率都是 1/3，损失为 ln3≈1.098612。

损失 L(w) 是 w 的函数。梯度（单参数时就是导数）描述当前附近 w 增加一点时，损失变化的速率，不是损失本身，也不是更新后的参数。中心差分给出近似：g≈[L(w+ε)-L(w-ε)]/(2ε)。两次都从原 w 出发，试探时不累计修改 w；分母 2ε 是两个试探位置的距离。

g<0 表示当前附近增加 w 会降低损失；g>0 则减小 w 有望降低损失。这是局部信息，不是任意大步都保证有效。梯度下降采用 w_new=w-ηg，η>0 为学习率。负梯度被减去时会增加参数，不应把“下降”理解为参数数值一定减小。

### 第一块：定义单位置损失

```python
import numpy as np  # 导入 NumPy，进行稳定的对数概率和数值计算。

def loss_at(weight):  # 给定一个可调参数，计算当前预测的损失。
    logits = np.array([weight, 0.0, 0.0], dtype=np.float64)  # 三个候选中只调节第 0 项的分数。
    shifted = logits - logits.max()  # 减去最大分数，保持指数计算稳定。
    log_probs = shifted - np.log(np.exp(shifted).sum())  # 直接得到三个候选的对数概率。
    return float(-log_probs[0])  # 正确标签固定为 0，返回其负对数概率。
```

本课使用 float64，减少微小差分中的舍入误差；并不修改已有 RNN 的 float32 参数。为独立演示保留这几行稳定损失公式，不需要通过特殊方式导入数字开头的旧脚本。

### 第二块：用两次试探计算数值梯度

```python
def numerical_gradient(weight, epsilon=1e-5):  # 用中心差分估计当前参数处的导数。
    assert epsilon > 0, "试探距离必须大于零"  # 分母不能为零，也不使用负试探距离。
    loss_plus = loss_at(weight + epsilon)  # 从原参数位置向正方向试探一步。
    loss_minus = loss_at(weight - epsilon)  # 从同一个原位置向负方向试探一步。
    return (loss_plus - loss_minus) / (2 * epsilon)  # 损失之差除以两个试探位置之间的距离。
```

epsilon 是估计斜率时的微小试探距离；learning_rate 是实际更新参数的步长系数，二者用途不同。epsilon 太大时近似误差可能更大，太小时又可能出现舍入误差，不是越小越好。

### 第三块：沿负梯度更新一次并核对

```python
def main():  # 定义演示入口，运行时观察一次参数更新。
    weight = 0.0  # 从三个候选分数相等的位置开始。
    learning_rate = 0.5  # 设置实际更新所用的正学习率。
    epsilon = 1e-5  # 设置计算数值梯度时的试探距离。
    before = loss_at(weight)  # 记录更新前的损失。
    gradient = numerical_gradient(weight, epsilon)  # 估计损失相对这个参数的变化速率。
    updated_weight = weight - learning_rate * gradient  # 沿负梯度方向更新参数。
    after = loss_at(updated_weight)  # 重新前向计算，检查参数改变后的损失。
    opposite_loss = loss_at(weight + learning_rate * gradient)  # 仅作对照，计算反方向步进后的损失。
    assert np.isfinite(gradient) and after < before < opposite_loss, "本例更新方向或损失不正确"  # 核对当前确定例子的方向，不代表所有大步长都有此保证。
    print("向正方向试探后的损失：", loss_at(weight + epsilon))  # 应略低于初始损失。
    print("向负方向试探后的损失：", loss_at(weight - epsilon))  # 应略高于初始损失。
    print("更新前参数：", weight)  # 初始为 0。
    print("数值梯度：", gradient)  # 当前约为 -0.666667。
    print("更新后参数：", updated_weight)  # 当前约为 0.333333。
    print("更新前损失：", before)  # 当前约为 1.098612。
    print("更新后损失：", after)  # 应小于初始损失。
    print("更新前正确答案概率：", np.exp(-before))  # 根据 loss=-ln(p) 还原单位置正确答案概率。
    print("更新后正确答案概率：", np.exp(-after))  # 当前应比 1/3 大。
    print("数值梯度与单步更新方向检查：通过")  # 上述断言通过后显示。

if __name__ == "__main__":  # 直接运行文件时才执行演示。
    main()  # 调用入口，不自动启动真正的 RNN 训练。
```

### 与真实 RNN 的联系

本例从参数到损失进行两次前向试探，用数值差分估计梯度，再执行一次梯度下降。没有执行反向传播。真实 RNN 有 719723 个参数，每个元素都有对应偏导数，某个参数数组的梯度应与该参数数组同形状。

若对全部参数都用中心差分，一次全量梯度约需 2×719723=1439446 次损失计算，远比用链式法则的反向传播昂贵。数值差分适合少量参数的理解和梯度检查，不适合作为真实模型的常规训练方式。后续会推导解析梯度，再对小模型选取少数元素与中心差分对照。

本次只是提高一条固定标签的概率，不说明模型对其他样本更好，也不是完整语料训练。待提交：完整代码和运行输出；口述题：若参数为 1.0、梯度为 -0.4、学习率为 0.1，更新后参数是多少，参数应增大还是减小？epsilon 和 learning_rate 分别做什么？

### 第十四课实操复查：通过

学习者提交的运行结果与参考一致：梯度约 -0.66666667，参数从 0 更新至约 0.33333333，损失从 1.09861229 降至 0.88915080，正确答案概率由 1/3 升至约 0.41100463。

老师检查实际脚本并重新运行；在 w=-3、0、2 三个位置，将数值梯度与本例的解析参考 p_correct-1 对照，均在容差内一致，按学习率 0.5 下降一步后损失均降低；epsilon=0 被拒绝。解析参考仅供老师复查，后续再让学习者推导，不能视为其已掌握。

学习者正确计算 1.0-0.1×(-0.4)=1.04，参数增大。学习者将 epsilon 称为“最小的变化量”，需修正：它是人为选定的微小试探距离，不是数学或计算机中存在的最小变化量；过大或过小都会影响数值梯度近似。learning_rate 是实际更新所乘系数，实际位移为 -learning_rate×gradient，不是每次都固定变化 learning_rate。

### 新问题：如何知道更新方向

中心差分中，2epsilon>0，所以梯度符号由 L(w+epsilon)-L(w-epsilon) 的符号决定。本次 L(w+epsilon)≈1.09860562 小于 L(w-epsilon)≈1.09861896，说明当前附近增加 w 会降低损失，梯度因此为负。公式 w_new=w-learning_rate×gradient 会自动将负梯度转成正位移，不需手写额外的方向判断。

对正梯度，当前附近增加参数会升高损失，负梯度更新使参数减小；对负梯度，更新使参数增大。零梯度说明一阶变化为零或很小，不足以单独证明达到最优。

一阶近似为 DeltaL≈g×Deltaw，选 Deltaw=-eta×g，则 DeltaL≈-eta×g²≤0，说明足够小步长下沿负梯度有下降趋势。它是局部近似，不保证任意学习率都下降；本课实际重新计算 after 是对当前这一步的验证。

真实模型有许多参数，每个元素有自己的偏导数，可同时出现正负号。普通梯度下降对每个元素分别做 theta_i←theta_i-eta×g_i，不是让所有参数统一增大或统一减小；后续优化器会再加入各自的更新规则。

当前记录为实操通过、更新算术正确、epsilon 定义已纠正、方向原理已补充讲解。下一课计划推导 softmax 与单标签交叉熵组合的解析梯度 p-q，并与少量数值差分对照，再逐层传播到输出权重和循环层。

### 第十四课后续讨论：方向、迭代与参数搜索

1. 学习者正确算出正梯度 0.4、学习率 0.1、参数 1 时更新为 0.96，但称“往 g>0 移动”。纠正：g>0 是梯度符号，实际位移为 -0.04，参数沿负梯度方向减小；梯度方向与普通梯度下降的更新方向相反。
2. 学习者问是否为同一参数计算很多次梯度后比较，选损失最小的值。解释：普通训练是在每个当前参数位置、当前批次上计算梯度，更新一步后再在新位置计算，不把历次梯度拿来挑选“最好梯度”。中心差分的两次前向仅用于估计一个当前偏导数。
3. 学习者追问“搜索候选参数”是什么。解释：尝试多个参数候选并比较损失属于参数搜索；有限候选中的最好值不保证全局最优，也可能漏掉候选范围之外或间隔之间的更好值。
4. 若 719723 个权重每个试 3 个值，穷举组合为 3^719723，不可行。固定其他参数、逐个搜索是坐标搜索的一类做法，虽然避免全部组合穷举，仍需要大量损失计算，且参数相互作用会使先前选出的值需要再调整。
5. 反向传播通过链式法则有效计算当前批次对全部参数的梯度，适合这里大量连续权重的训练；网格/随机搜索仍可用于少量学习率、隐藏维度等配置，并根据验证集选择，测试集不参与挑选。线搜索则可在已定方向上选择步长，无梯度方法也有适用目标，不能说搜索一概不能使用。

学习者表示了解并要求完整记录后进入下一课；记录为已讲解并确认理解，未据此推断其已能独立设计搜索算法或实现反向传播。

## 第 15 课：softmax 加交叉熵的解析梯度

状态：实操复查通过，学习者已完成 scripts/learn/09_softmax_gradient.py，标签变换复习题回答正确。本课使用三候选单位置示例，不修改 RNN 权重、不训练语料、不保存模型；推导理解仍可继续巩固。

### 推导

正确类别为 y，候选分数为 z，S=sum_k exp(z_k)，损失 L=-ln p_y=-z_y+ln S。

对某个候选分数 z_j 求偏导，其他分数暂时固定：

- -z_y 的偏导为 -1（j=y），否则为 0。
- ln S 的偏导为 (1/S)×exp(z_j)=p_j；这里用到了链式法则。
- 因而 dL/dz_j=p_j-1（j=y），否则为 p_j。用 one-hot 标签分布 q 合并写成 grad=p-q。

这是损失对原始分数 z 的梯度，不是对概率 p 的梯度，也不是直接对循环权重的梯度。错误候选也有梯度，因为它们通过 softmax 分母影响正确答案的概率。无需构造 V×V 的 softmax 雅可比矩阵。

候选为天/风/雨，目标为天：q=[1,0,0]。对分数 [0.15,-0.1,-0.3]，p≈[0.41383381,0.32229410,0.26387209]，梯度约 [-0.58616619,0.32229410,0.26387209]。普通负梯度更新会提高正确候选分数、降低另外两个分数。

### 第一块：计算单位置损失、概率和解析梯度

```python
import numpy as np  # 导入 NumPy，用 float64 检查微小梯度差异。

def loss_and_grad(logits, target):  # 计算一个位置的损失、概率以及损失对各原始分数的梯度。
    logits = np.asarray(logits, dtype=np.float64)  # 本演示统一使用双精度，不改已有 RNN 的参数类型。
    assert logits.ndim == 1 and logits.size > 0 and np.isfinite(logits).all(), "需要有限的一维分数"  # 本课每次只检查一个预测位置。
    assert isinstance(target, (int, np.integer)) and 0 <= target < logits.size, "正确标签越界或类型错误"  # 标签是一个候选编号。
    shifted = logits - logits.max()  # 先平移分数，保持数值稳定。
    log_probs = shifted - np.log(np.exp(shifted).sum())  # 直接计算所有候选的对数概率。
    probabilities = np.exp(log_probs)  # 根据对数概率恢复概率，供梯度公式使用。
    gradient = probabilities.copy()  # 梯度从 p 开始，复制以免改写概率数组。
    gradient[target] -= 1.0  # 正确类别减去 1，其他类别相当于减去 0，即 p-q。
    return float(-log_probs[target]), probabilities, gradient  # 返回标量损失、概率向量和同形状分数梯度。
```

### 第二块：每次只扰动一个分数，计算数值梯度

```python
def numerical_gradient(logits, target, epsilon=1e-5):  # 用中心差分独立核对每个分数的偏导。
    assert epsilon > 0, "试探距离必须大于零"  # 不允许分母为零。
    logits = np.asarray(logits, dtype=np.float64)  # 使用双精度进行差分。
    numerical = np.zeros_like(logits)  # 准备与分数向量同形状的数值梯度。
    for index in range(logits.size):  # 依次计算各个候选分数的偏导。
        plus = logits.copy()  # 正向试探始终从同一组原分数开始。
        minus = logits.copy()  # 反向试探也从同一组原分数开始。
        plus[index] += epsilon  # 只增加当前分数，其他分数固定。
        minus[index] -= epsilon  # 只减少当前分数，其他分数固定。
        plus_loss, _, _ = loss_and_grad(plus, target)  # 数值差分仅使用损失，忽略函数额外返回的结果。
        minus_loss, _, _ = loss_and_grad(minus, target)  # 计算反向试探的损失。
        numerical[index] = (plus_loss - minus_loss) / (2 * epsilon)  # 两个损失之差除以试探位置距离。
    return numerical  # 返回全部候选的数值偏导，不在试探过程中更新原分数。
```

若同时移动所有分数，得到的不是某一个坐标的偏导；所有候选同加一个数时损失不变，甚至可能错误得到全零“梯度”。所以每次只修改一个元素。

### 第三块：检查梯度并演示一次分数更新

```python
def main():  # 定义检查入口，独立验证本课公式。
    logits = np.array([0.15, -0.1, -0.3], dtype=np.float64)  # 设置天、风、雨三个候选的原始分数。
    target = 0  # 本次正确答案是第 0 项天。
    before, probabilities, gradient = loss_and_grad(logits, target)  # 根据 p-q 计算解析梯度。
    numerical = numerical_gradient(logits, target)  # 根据两侧损失计算独立的数值近似。
    max_error = np.max(np.abs(gradient - numerical))  # 显示两种方法的最大绝对差。
    assert np.allclose(gradient, numerical, rtol=1e-6, atol=1e-8), "解析与数值梯度不一致"  # 在双精度容差内核对梯度。
    assert np.isclose(gradient.sum(), 0.0, atol=1e-12), "梯度和不为零"  # 同时平移所有分数不改损失，因此这些偏导之和应为零。
    updated_logits = logits - 0.5 * gradient  # 只为观察方向，直接对三个分数走一个负梯度小步。
    after, updated_probabilities, _ = loss_and_grad(updated_logits, target)  # 重新计算演示更新后的损失和概率。
    assert after < before and updated_probabilities[target] > probabilities[target], "本例下降方向不正确"  # 核对当前例子的更新效果。
    print("模型概率：", probabilities)  # 当前约为 [0.41383381,0.32229410,0.26387209]。
    print("解析梯度：", gradient)  # 正确项为负，错误项为正。
    print("数值梯度：", numerical)  # 应与解析结果接近。
    print("最大绝对误差：", max_error)  # 应很小，具体尾数可能随平台变化。
    print("更新前损失：", before)  # 显示当前分数对应的损失。
    print("更新后损失：", after)  # 本例应小于更新前。
    print("解析梯度与数值差分检查：通过")  # 所有断言通过后显示。

if __name__ == "__main__":  # 直接运行本文件时启动演示。
    main()  # 调用入口，不改变现有 RNN 模型。
```

### 与平均损失、参数梯度的关系

本课只有一个位置，因此梯度是 p-q；若损失对 N=B×L 个有效位置取平均，则各位置分数梯度为 (p-q)/N，形状为 (B,L,V)。这个除数只应在对应计算链中应用一次，不能每层反向又重复除以 N。

真实 RNN 的 logits 是参数前向计算的结果，不是本课直接更新后保存的一组权重。本课直接更新 logits 只为验证方向。下一步需用链式法则，把分数梯度传给输出权重、输出偏移和隐藏状态，随后继续传回循环层。

计算 p-q 本身为 O(V)，中心差分核对全部 V 个分数约需 2V 次损失计算；本课 V=3，仅核对一个小例子，不能因此用数值差分训练完整 RNN。没有真实语料训练结果。

待提交：完整脚本、概率/两组梯度/误差输出，并回答：若 p=[0.2,0.7,0.1]，正确标签为第 0 项，单位置分数梯度是多少？普通梯度下降会让三个分数分别增大还是减小？实操通过后再更新结果、CI 并上传 GitHub。

### 第十五课补充讲解：从减法到梯度来源

学习者最初表示不理解，老师先用 p=[0.2,0.7,0.1]、q=[1,0,0] 逐项相减得到 [-0.8,0.7,0.1]，再说明减去学习率乘梯度会提高正确分数、降低其他分数。实际更新的是 logits，不是直接把位移加到概率上。

复习题 p=[0.6,0.3,0.1]、正确标签仍为第 0 项，学习者回答 [-0.6,0.3,0.1]。第一项应为 0.6-1=-0.4，正确结果为 [-0.4,0.3,0.1]；后两项正确。学习者继续问为什么概率减标签是梯度，因此本课暂停新代码，补充如下推导。

令三个原始分数为 a、b、c，S=exp(a)+exp(b)+exp(c)，正确项是 a。p0=exp(a)/S，L=-ln(p0)=-a+ln(S)。这是把损失直接写成原始分数的函数，不把概率当作与分数无关的常量。

- 对 a 求导：-a 给出 -1；ln(S) 通过链式法则给出 (1/S)×exp(a)=p0。因此 dL/da=p0-1。
- 对 b 求导：-a 与 b 无关，给出 0；ln(S) 给出 exp(b)/S=p1。因此 dL/db=p1。
- 对 c 同理：dL/dc=p2。

合并得到 [p0-1,p1,p2]=p-[1,0,0]。如果正确项换成其他编号，只有对应的 -z_y 项会贡献 -1，所以一般式为 p-q。标签不是可调参数，one-hot 只是把正确/错误候选两种求导情况统一表达。

注意：p-q 是 softmax 与单标签交叉熵组合对 logits 的梯度，不是“任何概率减标签都叫梯度”。如果直接对正确概率 p_y 求损失导数，得到 -1/p_y；还必须把概率对分数的依赖纳入链式法则，才能得到分数梯度 p-q。平均多个位置还需要相应除以有效位置数。

老师用 logits=ln([0.6,0.3,0.1]) 独立进行中心差分，结果与 [-0.4,0.3,0.1] 一致。只将正确分数增大 0.001，实际损失约减少 0.00039988，与一阶预测 -0.4×0.001=-0.0004 接近，说明该梯度确实描述当前附近的损失变化速率。

以上为当时补充讲解与算术纠正的记录，后续实操已提交并通过复查，详见下文。

### 第十五课复习题与实操复查：通过

对于 p=[0.6,0.3,0.1]、正确标签为第 1 项，学习者回答 q=[0,1,0]，梯度为 [0.6,-0.7,0.1]，正确。普通负梯度更新会降低第 0、2 项分数，提高正确的第 1 项分数。记录为已正确应用此公式，不据此推断完整链式法则或 RNN 反向传播已独立掌握。

学习者提交脚本运行输出，老师检查实际代码并重新运行，结果一致：解析和数值梯度均约 [-0.58616619,0.32229410,0.26387209]，最大绝对差约 4.1310e-12；演示分数更新后的损失由约 0.88229080 降至 0.64780622。

老师另外验证：

- 1、3、5 个候选、所有合法目标位置、等分数和随机分数下，解析梯度与中心差分均在容差内一致。
- 原始分数在梯度函数与数值试探后保持不变，每次差分只扰动当前元素。
- 梯度之和约为 0，统一平移全部分数不改变损失。
- 现有实现未修改 RNN 参数；两种梯度的差不是模型预测误差，也不表示真实语料训练已完成。
- GitHub 自动检查增加 09_softmax_gradient.py，持续核对该教学公式和数值差分。

本课代码与关于梯度来源、标签选择、方向以及中心差分的补充笔记一并提交。下一课计划用链式法则，把 logits 梯度传给输出权重、输出偏移和隐藏状态，明确它们各自的形状与跨位置累加方式。

## 第 16 课：输出层反向传播

状态：代码实操复查通过，学习者已在 scripts/learn/09_softmax_gradient.py 完成输出层反向和三组数值检查。口述题学习者表示不知道，形状和梯度累加原理补充见本课末尾。此课未改完整 RNN、未训练小说语料、未保存权重。

### 从分数梯度到参数梯度

输出层 Z=AW+b，其中 A 为隐藏状态，形状 (N,H)，W 为 (H,V)，b 为 (V,)。G=dL/dZ 与 Z 同形状 (N,V)。N 可以表示批次和位置展平后的数量，当前演示 N=1、H=2、V=3。

单个分数 Z[n,j]=sum_i A[n,i]W[i,j]+b[j]。链式法则要求上游损失变化速率乘本步局部变化速率，并对所有影响路径求和：

- dL/dW[i,j]=sum_n A[n,i]G[n,j]，矩阵写法 grad_weight=A.T@G。
- dL/db[j]=sum_n G[n,j]，代码 grad_bias=G.sum(axis=0)。共享偏移在每一行都用到，所以贡献相加，不再重复平均。
- dL/dA[n,i]=sum_j G[n,j]W[i,j]，矩阵写法 grad_hidden=G@W.T。

| 梯度 | 计算形状 | 返回形状 |
| --- | --- | --- |
| 权重梯度 | (H,N)@(N,V) | (H,V)，与 W 一致 |
| 偏移梯度 | (N,V) 沿 N 求和 | (V,)，与 b 一致 |
| 隐藏状态梯度 | (N,V)@(V,H) | (N,H)，与 A 一致 |

若 G 已由平均交叉熵得到，包含 1/N，则本层不要再除以 N。当前一位置 loss_and_grad 返回 p-q，无额外平均。

### 第一块：在 main 之前增加输出层的损失和反向函数

下列函数均在类外顶格。本文件本来就是独立教学脚本，保留已有 loss_and_grad 与 numerical_gradient：

```python
def output_loss(hidden, weight, bias, target):  # 计算单个位置经过输出层后的交叉熵。
    logits = hidden @ weight + bias  # 从 (1,H) 状态计算 (1,V) 分数。
    assert logits.shape[0] == 1, "本演示损失只处理一个位置"  # 复用已有单位置损失函数。
    loss, _, _ = loss_and_grad(logits[0], target)  # 取出这一行分数，计算标量损失。
    return loss  # 返回数值差分需要的损失值。

def output_backward(hidden, weight, grad_logits):  # 接收上游分数梯度，计算输出层的三组梯度。
    assert hidden.ndim == weight.ndim == grad_logits.ndim == 2, "需要二维状态、权重和分数梯度"  # 统一使用矩阵形状。
    assert hidden.shape[0] == grad_logits.shape[0] and hidden.shape[1] == weight.shape[0] and grad_logits.shape[1] == weight.shape[1], "反向传播形状不匹配"  # 核对 N、H、V 的对应关系。
    grad_weight = hidden.T @ grad_logits  # 汇总各位置对共享输出权重的贡献，得到 (H,V)。
    grad_bias = grad_logits.sum(axis=0)  # 汇总各位置对共享输出偏移的贡献，得到 (V,)。
    grad_hidden = grad_logits @ weight.T  # 将分数梯度沿输出权重传回隐藏状态，得到 (N,H)。
    return grad_weight, grad_bias, grad_hidden  # 参数梯度用于更新，状态梯度留给前面的层继续反传。
```

### 第二块：通用小数组数值检查

同样放在 main 之前顶格添加：

```python
def numerical_array_gradient(value, evaluate, epsilon=1e-5):  # 用给定的损失计算方法，逐元素检查一个小数组的梯度。
    assert epsilon > 0, "试探距离必须大于零"  # 防止无效差分距离。
    value = np.asarray(value, dtype=np.float64)  # 使用双精度进行微小扰动。
    gradient = np.zeros_like(value)  # 梯度形状与被检查的数组相同。
    for index in np.ndindex(value.shape):  # 依次取得 (行,列) 等下标，兼容向量和矩阵。
        plus = value.copy()  # 正向试探使用独立副本。
        minus = value.copy()  # 反向试探也使用独立副本。
        plus[index] += epsilon  # 每次只增加一个元素。
        minus[index] -= epsilon  # 每次只减少同一个元素。
        gradient[index] = (evaluate(plus) - evaluate(minus)) / (2 * epsilon)  # 调用传入的方法计算两侧损失。
    return gradient  # 返回数值近似，不改动原数组。
```

evaluate 是一个函数参数：收到候选数组后返回损失。下面用 lambda 指定候选数组要替换 W、b 或 hidden，其余保持固定。np.ndindex 对 (2,3) 依次给出 (0,0)、(0,1)、(0,2)、(1,0) 等位置，不是将所有元素一起扰动。

### 第三块：在原 main 末尾添加

整体缩进 4 个空格，保留原有全部检查：

```python
    hidden = np.array([[0.2, -0.1]], dtype=np.float64)  # 一个位置、两个隐藏分量，形状为 (1,2)。
    weight = np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]], dtype=np.float64)  # 三个候选的输出权重，形状为 (2,3)。
    bias = np.array([0.0, 0.1, 0.0], dtype=np.float64)  # 每个候选一个偏移。
    target = 0  # 正确答案仍为第 0 项。
    scores = hidden @ weight + bias  # 得到 [0.15,-0.1,-0.3]，但保留 (1,3) 形状。
    current_loss, _, score_gradient = loss_and_grad(scores[0], target)  # 先取得对三个分数的梯度。
    grad_weight, grad_bias, grad_hidden = output_backward(hidden, weight, score_gradient.reshape(1, -1))  # 恢复一行形状并计算三组梯度。
    numeric_weight = numerical_array_gradient(weight, lambda candidate: output_loss(hidden, candidate, bias, target))  # 只试探输出权重。
    numeric_bias = numerical_array_gradient(bias, lambda candidate: output_loss(hidden, weight, candidate, target))  # 只试探输出偏移。
    numeric_hidden = numerical_array_gradient(hidden, lambda candidate: output_loss(candidate, weight, bias, target))  # 只试探隐藏状态，不将其当作模型参数。
    assert np.allclose(grad_weight, numeric_weight, rtol=1e-6, atol=1e-8), "输出权重梯度不正确"  # 核对权重每个元素。
    assert np.allclose(grad_bias, numeric_bias, rtol=1e-6, atol=1e-8), "输出偏移梯度不正确"  # 核对每个偏移。
    assert np.allclose(grad_hidden, numeric_hidden, rtol=1e-6, atol=1e-8), "隐藏状态梯度不正确"  # 核对传给前层的梯度。
    updated_weight = weight - 0.1 * grad_weight  # 沿负梯度更新演示输出层权重。
    updated_bias = bias - 0.1 * grad_bias  # 沿负梯度更新演示输出层偏移。
    after_output_update = output_loss(hidden, updated_weight, updated_bias, target)  # 固定隐藏状态，重新前向计算。
    assert after_output_update < current_loss, "本例输出层更新没有降低损失"  # 验证这个具体小步的效果。
    print("输出权重梯度：", grad_weight)  # 应为 (2,3)，每个权重一个偏导。
    print("输出偏移梯度：", grad_bias)  # 应为 (3,)。
    print("隐藏状态梯度：", grad_hidden)  # 应为 (1,2)，传给循环层的接口。
    print("三组梯度最大误差：", np.max(np.abs(grad_weight - numeric_weight)), np.max(np.abs(grad_bias - numeric_bias)), np.max(np.abs(grad_hidden - numeric_hidden)))  # 显示各组数值检查误差。
    print("输出层更新前后损失：", current_loss, after_output_update)  # 这次更新的是产生分数的参数，不直接改 scores。
    print("输出层反向传播与数值检查：通过")  # 全部断言通过后显示。
```

### 形状与数值解释

本例分数梯度约 [-0.58616619,0.32229410,0.26387209]，W[0,0] 的梯度是 hidden[0,0]×G[0,0]=0.2×(-0.58616619)≈-0.11723324；W[1,0] 的梯度则是 (-0.1)×(-0.58616619)≈0.05861662。即使同属于正确候选一列，权重更新方向也可能不同，因为隐藏分量有不同符号。

grad_hidden 用于继续沿计算链向前层传播，不直接执行 hidden-=学习率×grad_hidden；真实隐藏状态由之前的模型参数算出，不是需要持久保存优化的独立参数。此处只更新小输出层 W、b，已有 RNN 不受影响。

完整 RNN 的 states=(B,L,H) 可先展平为 (B×L,H)，分数梯度同序展平为 (B×L,V) 后应用相同公式，再将状态梯度恢复为 (B,L,H)。形状不能凭转置死记，应由每个共享参数的影响路径和求和关系推导。

待提交：修改后的脚本、三组梯度与误差、更新前后损失。口述题：hidden=(4,2)、weight=(2,3)、grad_logits=(4,3) 时，三组梯度分别什么形状？为什么偏移梯度要沿第 0 维求和，而不是再除以 4？实操通过后更新笔记和 CI 并完整上传 GitHub。

### 第十六课实操复查：通过

学习者提交结果与参考一致：权重梯度 (2,3)、偏移梯度 (3,)、隐藏状态梯度 (1,2)，三组中心差分最大绝对差约为 8.12e-12、7.66e-12、6.15e-12；小输出层更新后损失从 0.88229080 降至 0.82904450。

老师检查实际代码并重新运行；另行构造四行状态 (4,2)、权重 (2,3)、偏移 (3,)，以四行平均交叉熵为目标核对三个数组的全部元素，解析与数值梯度均一致。测试确认原输入、权重、偏移及上游梯度未被反向函数或数值试探改写。本次未代改学习者实现，GitHub 自动检查继续执行当前教学脚本的全部断言。

### 口述题补充：每个原始元素对应一个偏导

学习者表示不知道。梯度回答“这个具体元素改变一点，损失怎样变”，所以每个需要求导的元素对应一个梯度数。weight 为 (2,3) 有 6 个元素，grad_weight 也为 (2,3)；bias 为 (3,) 有 3 个元素，grad_bias 为 (3,)；hidden 为 (4,2) 有 8 个元素，grad_hidden 为 (4,2)。梯度保持对应对象的形状，不把所有对象的维度拼接起来。

矩阵公式逐步核对：hidden.T 为 (2,4)，乘 grad_logits=(4,3) 得 (2,3)；grad_logits 沿 axis=0 将四行贡献合并，得 (3,)；grad_logits=(4,3) 乘 weight.T=(3,2) 得 (4,2)。

### 为什么偏移梯度求和，为什么不能重复平均

同一个 bias[j] 被加到每行的第 j 个分数上，不是每行拥有一套独立偏移。若 bias[0] 增加一点，四行第 0 项分数都会增加同样一点，因此损失通过四条路径变化，总贡献必须相加。axis=0 对应行维度，所以 grad_logits.sum(axis=0) 将四行合成各候选偏移的三个梯度。

损失若已定义为 L=(l0+l1+l2+l3)/4，那么上游 G[n,j]=dL/dZ[n,j] 已等于该位置未平均梯度除以 4。求共享偏移梯度时只需合并这些已经缩放的贡献，不再除 4。

例如某个偏移的四个未平均贡献为 [0.4,0.8,-0.2,0.6]，传入本层的 G 对应一列已为 [0.1,0.2,-0.05,0.15]。求和得到 0.4，等于原贡献总和 1.6/4；若再除 4 则变为 0.1，相当于把整条损失错误地又缩小四倍。这里的 sum 是合并共享参数的多条作用路径，不是在定义新的损失平均。

如果上游传入的是总损失梯度而不是平均损失梯度，那也是对那个已定义目标求和，不能在中间层擅自改变目标；应从整体损失定义统一决定归一化。

当前记录为实操通过、概念已补充讲解，不提前视为独立掌握。先做单偏移两行贡献相加的复习，再进入循环层 tanh 的局部梯度和跨时间反传。

### 第十六课共享偏移复习题

对已经包含平均因子的两行上游贡献 0.3 和 -0.1，学习者回答共享偏移总梯度为 0.2，正确。记录为该复习题通过，不据此认定所有矩阵梯度公式已经独立掌握。

## 第 17 课：tanh 与循环层单步反向传播

状态：实操复查通过，学习者已实现循环单步反向及四组数值检查，tanh 标量梯度题回答正确；逐元素乘法原因需补充讲解。不改完整 RNN、不跨时间反传、不训练小说、不更新权重。

### 一步循环的计算链

先计算激活前数值 a=current+previous@W_h+b_h，再令 hidden=tanh(a)，之后经过输出层与交叉熵得到损失。

顺序为：当前字符向量和上一步状态 → 激活前数值 → tanh 后的状态 → 输出层 → 损失。反向按相反顺序传递，先复用输出层得到 G_h=dL/dhidden，然后穿过 tanh，再穿过加法和循环矩阵。

### tanh 导数与逐元素乘法

单个数 h=tanh(a)，导数 dh/da=1-h²；所以 G_a=G_h*(1-hidden²)。这里 * 为逐元素乘法，不是矩阵乘法，每个激活分量只依赖同位置的激活前分量。

导数可从指数定义推导：令 u=exp(2a)，tanh(a)=(u-1)/(u+1)，商法则给出导数 4u/(u+1)²；而 1-[(u-1)/(u+1)]² 也等于 4u/(u+1)²。因此已保存 h 时无需重新计算指数。

手算 h=0.8、G_h=0.5，则局部导数为 1-0.8²=0.36，G_a=0.5×0.36=0.18。h 接近 ±1 时局部导数很小，跨很多步相乘时可能削弱梯度，这是后续讨论长期依赖的一个原因，不代表所有循环网络梯度一定变小。

### 其余单步梯度

对 a=current+previous@W_h+b_h：

- grad_current=G_a：当前向量直接相加，局部导数为 1。
- grad_previous=G_a@W_h.T：将梯度沿循环连接传给前一时刻。
- grad_recurrent=previous.T@G_a：累积各行对共享循环权重的贡献。
- grad_hidden_bias=G_a.sum(axis=0)：共享偏移贡献沿行相加。

current、previous、hidden、G_h、G_a 均为 (B,H)，W_h 与其梯度为 (H,H)，隐藏偏移及其梯度为 (H,)。本课 B=1、H=2，所有检查用 float64，不改变已有 RNN 的 float32 配置。

### 第一块：在原 main 前顶格添加三个函数

```python
def recurrent_step(current, previous, weight, bias):  # 计算一个时间步，不处理整段序列。
    return np.tanh(current + previous @ weight + bias)  # 把当前向量、上一步状态和偏移结合，再逐元素激活。

def recurrent_step_loss(current, previous, recurrent_weight, hidden_bias, output_weight, output_bias, target):  # 将单步循环接到已有输出层损失。
    hidden = recurrent_step(current, previous, recurrent_weight, hidden_bias)  # 得到当前时间步的状态。
    return output_loss(hidden, output_weight, output_bias, target)  # 复用上一课单位置损失，不改变其他目标定义。

def recurrent_backward(previous, weight, hidden, grad_hidden):  # 将当前状态的上游梯度继续传过一个循环步。
    assert previous.ndim == 2 and previous.shape == hidden.shape == grad_hidden.shape, "状态与上游梯度形状不一致"  # 每行每个分量都要有对应梯度。
    assert weight.shape == (hidden.shape[1], hidden.shape[1]), "循环权重形状错误"  # 本模型输入和隐藏维度都为 H。
    grad_activation = grad_hidden * (1.0 - hidden ** 2)  # 梯度乘 tanh 局部导数，必须逐元素计算。
    grad_current = grad_activation.copy()  # 当前字符向量直接相加，收到相同数值的梯度。
    grad_previous = grad_activation @ weight.T  # 将梯度传回上一步状态，得到 (B,H)。
    grad_weight = previous.T @ grad_activation  # 汇总各行贡献，得到 (H,H) 的循环权重梯度。
    grad_bias = grad_activation.sum(axis=0)  # 汇总共享隐藏偏移的梯度，得到 (H,)。
    return grad_current, grad_previous, grad_weight, grad_bias  # 返回四组梯度，暂时不更新任何数组。
```

这里 hidden 是 tanh 后的状态，不能把 previous 误代入导数 1-hidden²；grad_hidden 是上游损失对状态的梯度，不是 hidden 本身。权重梯度要用对应这一步的 previous，而不是当前 hidden。

### 第二块：在原 main 末尾添加单步检查

保留上一课全部代码与变量。weight、bias、target 继续指上一段的小输出层参数；新变量 recurrent_weight 和 hidden_bias 指循环层，不能混淆。整体缩进 4 个空格：

```python
    current_vector = np.array([[0.1, 0.2]], dtype=np.float64)  # 当前字符的两维向量。
    previous_hidden = np.array([[0.2, -0.1]], dtype=np.float64)  # 上一步的两维状态，不要求本课重算它的来源。
    recurrent_weight = np.array([[0.5, 0.2], [-0.3, 0.4]], dtype=np.float64)  # 两维循环权重。
    hidden_bias = np.zeros(2, dtype=np.float64)  # 隐藏偏移，与上一课输出层的 bias 区分。
    step_hidden = recurrent_step(current_vector, previous_hidden, recurrent_weight, hidden_bias)  # 得到 tanh([0.23,0.2])。
    step_scores = step_hidden @ weight + bias  # 使用上一课输出层参数得到候选分数。
    step_loss, _, step_score_gradient = loss_and_grad(step_scores[0], target)  # 先计算损失和对分数的梯度。
    _, _, step_hidden_gradient = output_backward(step_hidden, weight, step_score_gradient.reshape(1, -1))  # 先经过输出层，得到当前状态的上游梯度。
    grad_current, grad_previous, grad_recurrent, grad_hidden_bias = recurrent_backward(previous_hidden, recurrent_weight, step_hidden, step_hidden_gradient)  # 再经过 tanh 和循环连接。
    numeric_current = numerical_array_gradient(current_vector, lambda candidate: recurrent_step_loss(candidate, previous_hidden, recurrent_weight, hidden_bias, weight, bias, target))  # 只试探当前字符向量。
    numeric_previous = numerical_array_gradient(previous_hidden, lambda candidate: recurrent_step_loss(current_vector, candidate, recurrent_weight, hidden_bias, weight, bias, target))  # 只试探上一步状态。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: recurrent_step_loss(current_vector, previous_hidden, candidate, hidden_bias, weight, bias, target))  # 只试探循环权重。
    numeric_hidden_bias = numerical_array_gradient(hidden_bias, lambda candidate: recurrent_step_loss(current_vector, previous_hidden, recurrent_weight, candidate, weight, bias, target))  # 只试探隐藏偏移。
    checks = [("当前向量", grad_current, numeric_current), ("上一步状态", grad_previous, numeric_previous), ("循环权重", grad_recurrent, numeric_recurrent), ("隐藏偏移", grad_hidden_bias, numeric_hidden_bias)]  # 统一组织四组检查。
    for name, analytic, numeric in checks:  # 依次核对每个对象的梯度。
        assert analytic.shape == numeric.shape and np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8), name + "梯度不正确"  # 对照完整数组而非单个元素。
        print(name + "梯度：", analytic)  # 显示该对象的解析梯度。
        print(name + "最大绝对误差：", np.max(np.abs(analytic - numeric)))  # 显示中心差分核对误差。
    print("当前状态：", step_hidden)  # 应约为 [[0.22602835,0.19737532]]。
    print("本例单步损失：", step_loss)  # 只用于核对固定例子，不代表真实模型测评。
    print("tanh 与循环单步反向检查：通过")  # 四组数值检查通过后显示。
```

数值差分每次从候选 current/previous/weight/bias 重新计算当前 hidden、分数和损失，不能把固定 step_hidden 再用进数值检查，否则会漏掉要验证的依赖路径。

### 单步与跨时间的边界

grad_previous 是传给前一个时间步的接口，不直接将 previous 当作独立参数更新。grad_current 后面需要累加到对应字符表示表行；重复字符的多次贡献也要累加，尚未在本课实现。

本课 grad_hidden 来自当前输出层；完整序列中还可能收到后一个时间步的回传，进入 tanh 之前必须合并这些路径，之后按时间反向累加共享参数梯度。不能仅做当前步输出梯度就宣称完整 RNN 反向传播正确。

本课局部反向的主要矩阵计算约 O(BH²)，没有新增参数，也没有运行更新。待提交：修改后的脚本、四组梯度和误差，并回答：某个 tanh 输出 h=0.8、上游梯度为 0.5，则激活之前的梯度是多少？为什么 grad_hidden*(1-hidden²) 用逐元素乘法，而不是 @ 矩阵乘法？完成后记录结果、更新 CI 并完整上传。

### 第十七课实操复查：通过

学习者提交的四组梯度、误差、当前状态和损失均与参考一致。老师检查实际脚本并重新运行：四组中心差分误差约为 1.99e-11、5.77e-12、5.05e-12、1.99e-11，当前状态约 [[0.22602835,0.19737532]]，单步损失约 1.06044782。

老师另行用一行和三行随机状态、任意固定上游线性目标检查四组数组梯度，形状与数值全部正确；前后输入、权重、偏移、状态及上游梯度均未被改写。未改学习者计算实现，只清理行尾空白。GitHub 自动检查继续运行已有演示中的 tanh 与循环单步断言。

### 口述题：标量计算正确，逐元素依赖继续解释

学习者回答激活前梯度为 0.18，正确：0.5×(1-0.8²)=0.18。对为什么使用逐元素乘法，学习者仍需解释，以下只记为补充讲解，不提前认定掌握。

单独看激活前向量 a=[a0,a1] 到激活后状态 h=[tanh(a0),tanh(a1)] 这一段：h0 只依赖 a0，h1 只依赖 a1，没有分量之间的混合。因此 dL/da0=G_h0×(1-h0²)，dL/da1=G_h1×(1-h1²)，每个上游梯度只乘对应位置的局部导数。

例如 h=[0.8,0.3]，G_h=[0.5,-0.2]，局部导数为 [0.36,0.91]，结果 G_a=[0.18,-0.182]。得到的梯度形状与 h 一样，不把不同分量相加。

完整链式法则也能写为行向量 G_h 乘一个对角雅可比矩阵 diag(1-h²)，非对角项全零；数值上与逐元素乘法一致，但构造对角矩阵会多占空间并做多余运算。本课不能直接把两个 (B,H) 数组做 @；即使尺寸偶然允许，也不表示符合这段计算的依赖。

前面的线性连接 previous@weight 本来就把多个状态分量混合到每个结果分量，所以反向用矩阵乘法汇总相应路径。选择 * 还是 @ 要从前向依赖推导，不能按“反向传播一律使用哪一种”记忆。这里讨论的是 a→tanh(a)，不是说整个循环状态不依赖其他历史分量。

当前记录为代码实操与标量题通过、逐元素原因已补充，待复习：只改变 a0、保持 a1 不变时，tanh(a1) 会不会变化？之后再串联两个时间步，合并当前输出和后续状态传回的梯度。

### 第十七课口述复核：通过

学习者回答：“不会变化，因为是相互独立的分量。”回答正确：在 a→tanh(a) 这一段，每个输出只依赖同一位置的输入，所以只改变 a0、保持 a1 不变，h1=tanh(a1) 不变。这里的独立指这一步计算的依赖关系，不是声称各个隐藏分量在统计上独立，也不是说前面的循环矩阵不会混合分量。

因此反向传播时，每个位置的上游梯度只乘该位置的局部导数：grad_activation = grad_hidden * (1-hidden**2)。本课代码实操、标量导数计算和逐元素依赖解释均已通过。下一课再串联两个时间步，学习合并当前输出与后续状态传回的梯度。

## 第十八课：两个时间步的反向传播与共享梯度累加

状态：老师已预检以下练习；待学习者亲自输入、运行并回答问题。未修改学习者脚本，不代表学习者已完成本课。继续在 scripts/learn/09_softmax_gradient.py 中练习，保留上一课全部内容。

### 先画清依赖关系

初始状态 h0，本次两个输入向量 x1、x2。这里 x 是已经查表得到的向量，不是字符编号；h0 沿用 previous_hidden，是给定初始状态，不要求全零。

h1=tanh(x1+h0@W_h+b_h)
h2=tanh(x2+h1@W_h+b_h)
z1=h1@W_o+b_o，z2=h2@W_o+b_o
L=(L1+L2)/2

h1 有两条通向损失的路径：h1→z1→L1→L，以及 h1→h2→z2→L2→L。h2 只有第二处输出这一条后续路径。本课只有两步，没有第三步损失。

先由输出层得到 direct1=(p1-q1)/2 @ W_o.T、direct2=(p2-q2)/2 @ W_o.T（实际代码先将分数梯度恢复为行矩阵）。第二步激活前梯度 A2=direct2*(1-h2²)，传回第一状态的梯度 future=A2@W_h.T。于是第一状态总梯度 G_h1=direct1+future，第一步激活前梯度 A1=G_h1*(1-h1²)。因此先反传第二步，再反传第一步，这就是跨时间反向传播 BPTT 的两步版本。

相加来自多元链式法则：若 L 通过 u(h1) 和 v(h1) 两条路径依赖 h1，dL/dh1 就是两条路径贡献之和。不能只保留当前输出路径，也不能把两个梯度相乘或再求平均。损失的 /2 已包含在各自输出梯度里，后续只按链式法则累加。

同一 W_h 在两个时间步重复使用，所以 dL/dW_h=h0.T@A1+h1.T@A2；隐藏偏移梯度为 A1.sum(axis=0)+A2.sum(axis=0)。输出权重与输出偏移也分别累加两个位置的贡献。第一项已通过 A1 包含未来的影响，再加第二次使用权重的直接贡献是共享参数链式求导，不是重复计算。

### 第一块：在 main 前添加一个辅助函数

此函数的正确标签固定为第一位置 0、第二位置 1，仅用于本课玩具例子。数值差分必须从候选输入或候选共享权重重新计算两步，不复用固定的旧状态。

```python
def two_step_loss(first, second, initial, recurrent_weight, hidden_bias, output_weight, output_bias):  # 重新计算两步的平均损失，供数值差分检查。
    first_hidden = recurrent_step(first, initial, recurrent_weight, hidden_bias)  # 从初始状态计算第一步状态。
    second_hidden = recurrent_step(second, first_hidden, recurrent_weight, hidden_bias)  # 第一状态参与第二步计算。
    first_loss = output_loss(first_hidden, output_weight, output_bias, 0)  # 本例第一个位置的正确编号固定为 0。
    second_loss = output_loss(second_hidden, output_weight, output_bias, 1)  # 本例第二个位置的正确编号固定为 1。
    return (first_loss + second_loss) / 2.0  # 两个预测位置的损失取平均。
```

### 第二块：在原 main 末尾添加检查

保留原 main 中 current_vector、previous_hidden、recurrent_weight、hidden_bias、weight、bias。weight 和 bias 仍是旧例子的输出参数，不使用 updated_weight 或 updated_bias。以下代码整体缩进四个空格，放在原来的最后一个 print 后面、文件末尾的 if __name__ 判断之前。

```python
    second_vector = np.array([[-0.2, 0.3]], dtype=np.float64)  # 新增第二步的输入向量，第一步沿用 current_vector。
    first_hidden = recurrent_step(current_vector, previous_hidden, recurrent_weight, hidden_bias)  # 从给定初始状态计算第一步。
    second_hidden = recurrent_step(second_vector, first_hidden, recurrent_weight, hidden_bias)  # 使用第一步状态计算第二步。
    first_loss, _, first_score_grad = loss_and_grad((first_hidden @ weight + bias)[0], 0)  # 第一位置的正确编号为 0。
    second_loss, _, second_score_grad = loss_and_grad((second_hidden @ weight + bias)[0], 1)  # 第二位置的正确编号为 1。
    mean_loss = (first_loss + second_loss) / 2.0  # 本例优化的是两个位置的平均损失。
    first_out_weight_grad, first_out_bias_grad, first_direct = output_backward(first_hidden, weight, first_score_grad.reshape(1, -1) / 2.0)  # 第一路输出梯度除以位置数，只除一次。
    second_out_weight_grad, second_out_bias_grad, second_direct = output_backward(second_hidden, weight, second_score_grad.reshape(1, -1) / 2.0)  # 第二路输出梯度也只除一次。
    second_input_grad, from_second, second_recurrent_grad, second_hidden_bias_grad = recurrent_backward(first_hidden, recurrent_weight, second_hidden, second_direct)  # 先反传第二步，取得传给第一步状态的梯度。
    first_total = first_direct + from_second  # 第一状态同时影响两处损失，必须合并两条路径。
    first_input_grad, initial_grad, first_recurrent_grad, first_hidden_bias_grad = recurrent_backward(previous_hidden, recurrent_weight, first_hidden, first_total)  # 合并后再通过第一步的 tanh 和循环连接。
    total_recurrent_grad = first_recurrent_grad + second_recurrent_grad  # 同一循环权重在两步使用，其梯度贡献相加。
    total_hidden_bias_grad = first_hidden_bias_grad + second_hidden_bias_grad  # 隐藏偏移也是两步共享的参数。
    total_output_weight_grad = first_out_weight_grad + second_out_weight_grad  # 输出权重在两处预测中的贡献也要相加。
    total_output_bias_grad = first_out_bias_grad + second_out_bias_grad  # 输出偏移同样汇总两个位置的贡献。
    numeric_first = numerical_array_gradient(current_vector, lambda candidate: two_step_loss(candidate, second_vector, previous_hidden, recurrent_weight, hidden_bias, weight, bias))  # 改变第一输入后重算完整两步，检查跨步路径。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: two_step_loss(current_vector, second_vector, previous_hidden, candidate, hidden_bias, weight, bias))  # 每次试探同一个共享权重，两步都使用候选值。
    assert np.allclose(first_input_grad, numeric_first, rtol=1e-6, atol=1e-8), "第一输入梯度不正确"  # 漏掉第二步传回的贡献会导致此检查失败。
    assert np.allclose(total_recurrent_grad, numeric_recurrent, rtol=1e-6, atol=1e-8), "共享循环权重梯度不正确"  # 覆盖其中一步贡献会导致此检查失败。
    print("两步平均损失：", mean_loss)  # 应约为 0.90833260。
    print("第一状态的直接梯度：", first_direct)  # 来自第一个位置的输出层。
    print("第二步传回的梯度：", from_second)  # 来自第二个位置，经循环连接传回。
    print("第一状态的总梯度：", first_total)  # 这才是第一步 tanh 接收的上游梯度。
    print("第一输入梯度：", first_input_grad)  # 应约为 [[-0.50547377,0.27042386]]。
    print("共享循环权重总梯度：", total_recurrent_grad)  # 显示两次使用共同产生的梯度。
    print("第一输入最大绝对误差：", np.max(np.abs(first_input_grad - numeric_first)))  # 显示第一输入的数值核对误差。
    print("共享权重最大绝对误差：", np.max(np.abs(total_recurrent_grad - numeric_recurrent)))  # 显示共享权重的数值核对误差。
    print("两步反向传播与梯度累加检查：通过")  # 两项断言都通过后显示。
```

### 如何理解形状、参数和计算量

本例 B=1、H=2、V=3：两个输入和各状态均为 (1,2)，循环权重及其梯度为 (2,2)，隐藏偏移及梯度为 (2,)，输出权重及梯度为 (2,3)，输出偏移及梯度为 (3,)。first_direct、from_second、first_total 都为 (1,2)，所以对应位置相加后形状不变。initial_grad 是传向给定初始状态的梯度，本课不更新初始状态；两个输入梯度后续才会用于表示表行的梯度累加。

增加时间步不会增加参数：两步共享 W_h、b_h、W_o、b_o。本演示这四组参数共 4+2+6+3=15 个数，独立给定的输入向量与初始状态不算这四组可训练参数。完整字符 RNN 还要加表示表，模型总参数仍是上一阶段的 719723 个。

一般 B 个窗口、T 个位置、隐藏维 H、词表 V 时，循环和输出层前向/反向的主要计算量均为 O(BT(H²+HV))；反向常数开销与前向不同。保存全部状态需 O(BTH) 空间，若同时保存全部分数或概率，还需 O(BTV) 空间。这是数量级估计，不是实测运行时间。教学用逐参数中心差分会反复前向，不作为正式训练算法。

### 运行与提交

运行 python scripts/learn/09_softmax_gradient.py，原有检查也应继续通过。老师在内存中组合上述代码预检，未替学习者写入脚本；并对两个输入、初始状态和四组共享参数共七组数组逐元素差分核对通过，故意漏掉未来路径会被检查识别。

预期新输出包括：平均损失约 0.9083325958；第一状态直接梯度 [[-0.44845625,0.36866788]]，第二步传回梯度 [[-0.08423192,-0.08728206]]，总梯度 [[-0.53268818,0.28138582]]；第一输入梯度 [[-0.50547377,0.27042386]]；共享循环权重总梯度约 [[-0.11520967,-0.00582196],[0.03822178,-0.07935490]]。差分误差通常在 1e-11 左右，尾数允许平台差异，以断言通过为准。

待提交：新增输出和两个问题的回答。问题一：第一状态从当前输出收到 0.3、从第二步收到 -0.1，这两项已经包含平均损失的缩放，那么合并后是多少？问题二：为什么共享循环权重的梯度要相加，不能只保留最后一个时间步的梯度？完成后复查实际脚本、记录结果并上传 GitHub。

### 第十八课实操复查：通过，参数共享原因继续讲解

老师检查学习者实际脚本并重新运行，旧检查和新增两步检查全部通过。两步平均损失 0.90833259577928；第一状态直接梯度 [[-0.44845625,0.36866788]]，未来传回梯度 [[-0.08423192,-0.08728206]]，合并 [[-0.53268818,0.28138582]]；第一输入梯度 [[-0.50547377,0.27042386]]，共享循环权重总梯度 [[-0.11520967,-0.00582196],[0.03822178,-0.07935490]]。第一输入及共享循环权重的差分最大误差分别为 1.083677592106369e-11、7.809538606073918e-12。仅清理脚本行尾空白，清理前后语法树一致，未改计算逻辑。已有 CI 会运行该脚本，无需增加重复任务。

口述第一题 0.3+(-0.1)=0.2 正确；第二题学习者问为什么共享参数要累加，以下为补充讲解，尚未视为已掌握。

用一个更小的计算图说明共享：h1=w，h2=h1*w，暂时令演示目标 L=h2，所以 L=w*w。这里刻意省去 tanh 和交叉熵，仅检查同一参数出现两次时的链式法则，不代替真实模型损失。w=2 时 L=4；改变共享 w，两处都同时改变，导数 dL/dw=2w=4。

若只计算最后一次乘法中显式 w 的贡献，把 h1 当成不变，会得到 h1=2，漏掉另一份 2。漏掉的路径是 w→h1→h2→L，其贡献为 (dL/dh2)*(dh2/dh1)*(dh1/dw)=1*w*1=2。直接路径 w→h2→L 贡献为 1*h1=2，总计 4。也可临时区分两个使用位置 u、v，令 F(u,v)=u*v，再同时代入 u=w、v=w，得到 dL/dw=∂F/∂u+∂F/∂v；两个独立参数才会分别保留两份梯度，共享参数只有一个最终梯度。

回到 RNN：a1=x1+h0@W_h+b_h，a2=x2+h1@W_h+b_h。每个 W_h 元素在两步是同一个数；更新时两处一起变。所以 dL/dW_h=h0.T@A1+h1.T@A2，代码中 total_recurrent_grad=first_recurrent_grad+second_recurrent_grad。A1 已包含后续损失经 h1 传回的影响，第二项对应 W_h 在第二步的直接使用，两项属于不同路径，不是把同一贡献算两次。即使只取最后位置损失，也可能同时有这两条路径，不能把“只保留最后一次使用的局部贡献”和“仅优化最后位置损失”混为一谈。

若要平均多个预测位置的损失，在输出梯度处已除以位置数，本处累加不再除一次。收集完同一参数在所有使用位置的梯度后，才统一执行一次参数更新，不能在两步反传之间修改共享权重。

待口述复核：上述 L=w*w、w=3 时，两个使用位置分别贡献多少梯度，总梯度是多少？本课实操与第一题已通过，第二题待复核。

### 共享参数小例子追问：总梯度正确，分项继续说明

学习者回答总梯度为 6，正确，并追问两个使用位置分别贡献多少。两个位置各贡献 3：暂时写 F(u,v)=u*v，则 ∂F/∂u=v=3、∂F/∂v=u=3；共享时 u=w、v=w，所以 dL/dw=3+3=6。u、v 只是区分同一参数的两次使用，不是把模型改为两个独立参数。

在 h1=w、L=h1*w 的计算图中，最后乘法里的直接路径贡献为 h1=3；经 h1 的间接路径贡献为 w*(dh1/dw)=3*1=3。这是这个对称例子恰好相等，真实 RNN 中两步的贡献通常不同，甚至可能符号相反。此处记录为总梯度回答正确、分项已解释，不将追问本身记为已独立掌握分项求导。

## 第十九课：用反向时间循环处理整段序列

状态：老师已在内存中预检本课代码，待学习者手敲、运行和回答问题。没有替学习者修改脚本。上一课实操及总梯度计算通过，两个使用位置的分项贡献已补充讲解；本课在反向循环中继续巩固累加原因，不提前记作全部独立掌握。

继续在 scripts/learn/09_softmax_gradient.py 添加代码，保留原内容。目标是将上一课手写的第二步、第一步反传改成任意正整数长度 T 的循环。前向按 0 到 T-1 计算；反向按 T-1 到 0 计算。每一步先 total=direct[:,time,:]+future，再通过 tanh 和循环连接，最后把本步共享参数贡献累加进总梯度。

### 数学与变量含义

direct 是各位置输出层对状态的直接梯度，已包含平均损失的缩放；future 是所有后续步骤沿循环路径传回当前状态的合计梯度。它不只是下一处输出层的直接梯度：下一步反传已经合并了更晚位置的影响。

以下用 h0 表示初始状态，ht 表示第 t 个真实时间步：G_ht=direct_t+future_t；A_t=G_ht*(1-ht²)；传给前态的 future_(t-1)=A_t@W_h.T；共享参数梯度累加 h_(t-1).T@A_t 以及 A_t.sum(axis=0)。Python 下标从零开始，因此 states[:,0,:] 是数学上的 h1，而不是初始状态 h0。

最后一个时间步之后没有其他损失，所以 future 初始为零。每次循环中 future 应替换为本步传向前态的新梯度，不能在旧值上继续 +=；旧值已经用于构成 total，再加会重复或串错时间位置。共享参数梯度则必须 +=，因为同一参数在各步重复使用。

### 第一块：main 前添加整段前向函数

```python
def sequence_hidden(vectors, initial, weight, bias):  # 按时间正向计算并保存全部状态。
    assert vectors.ndim == 3 and vectors.shape[1] > 0, "输入需要非空时间维"  # 输入形状为 (B,T,H)。
    assert initial.shape == (vectors.shape[0], vectors.shape[2]), "初始状态形状不对"  # 每条序列各有一个初始状态。
    states = np.zeros_like(vectors)  # 每个输入位置对应一个同维度状态。
    hidden = initial.copy()  # 从给定初始状态开始，不修改原数组。
    for time in range(vectors.shape[1]):  # 按 0、1、2 等顺序向前计算。
        hidden = recurrent_step(vectors[:, time, :], hidden, weight, bias)  # 复用已经验证的单步计算。
        states[:, time, :] = hidden  # 保存本步状态，反向传播时还要用到。
    return states  # 返回 (B,T,H) 的状态数组。
```

这里 vectors 已经是字符向量，形状 (B,T,H)。本模型直接把输入向量加到隐藏状态，所以输入向量维度和隐藏维度相同。不要把字符编号数组传入这个函数。

### 第二块：main 前添加反向时间循环

```python
def sequence_backward(states, initial, weight, direct):  # 将各位置输出层的梯度沿时间反向传播。
    assert states.ndim == 3 and states.shape == direct.shape and states.shape[1] > 0, "状态与直接梯度形状不对"  # 每个状态必须有对应的直接梯度。
    assert initial.shape == (states.shape[0], states.shape[2]), "初始状态形状不对"  # 初始状态形状为 (B,H)。
    grad_vectors = np.zeros_like(states)  # 保存对每个输入向量的梯度。
    grad_weight = np.zeros_like(weight)  # 共享循环权重的总梯度从零开始。
    grad_bias = np.zeros(states.shape[2], dtype=states.dtype)  # 共享隐藏偏移的总梯度从零开始。
    future = np.zeros_like(initial)  # 最后位置没有更晚的步骤，未来贡献从零开始。
    for time in reversed(range(states.shape[1])):  # 按 T-1、T-2 到 0 的顺序反传。
        previous = initial if time == 0 else states[:, time - 1, :]  # 第零步的前态必须取 initial，不能取下标 -1。
        total = direct[:, time, :] + future  # 合并当前输出路径与后续步骤传回的梯度。
        grad_current, future, step_weight, step_bias = recurrent_backward(previous, weight, states[:, time, :], total)  # 反传本步，并取得传向更早一步的梯度。
        grad_vectors[:, time, :] = grad_current  # 不同位置的输入梯度分别保存。
        grad_weight += step_weight  # 同一个循环权重的各次使用贡献要累加。
        grad_bias += step_bias  # 同一个隐藏偏移的各次使用贡献也要累加。
    return grad_vectors, future, grad_weight, grad_bias  # 循环结束时 future 就是初始状态的梯度。
```

必须保存前向状态，因为每一步 tanh 的导数依赖本步状态，权重梯度依赖本步前态。time=0 要单独取 initial；否则 Python 的 -1 会错误取最后一个真实状态。grad_vectors 每个位置独立赋值；若多个位置来自同一个字符的查表，稍后向表示表回传时才需要按字符编号汇总，本课尚未实现该部分。

### 第三块：main 前添加数值检查用损失

```python
def sequence_loss(vectors, initial, recurrent_weight, hidden_bias, output_weight, output_bias, targets):  # 重算整段序列的平均损失，供数值核对。
    assert vectors.shape[0] == 1 and targets.shape == (vectors.shape[1],), "本例损失只演示一条序列"  # 单位置损失函数暂时逐步调用。
    states = sequence_hidden(vectors, initial, recurrent_weight, hidden_bias)  # 每次试探都重新计算整段状态。
    losses = [output_loss(states[:, time, :], output_weight, output_bias, int(targets[time])) for time in range(states.shape[1])]  # 计算各位置的损失。
    return float(np.mean(losses))  # 对全部时间位置取平均。
```

前向和循环反向支持批大小 B；这里损失辅助函数和下面的输出层演示仍限定 B=1，避免同时增加新概念。因此输出分数梯度除以 T；以后处理一般批次平均损失时应除以实际 B*T。这里不收集输出参数梯度，输出参数梯度的跨位置累加上一课已有实现，本课专注循环层，尚未整合完整模型训练。

### 第四块：原 main 的末尾添加三步检查

保留旧例子变量 previous_hidden、recurrent_weight、hidden_bias、weight、bias；不要使用 updated_weight 或 updated_bias。下面整体缩进四个空格，放在原末尾 print 后、文件末尾 if __name__ 判断前。

```python
    vectors = np.array([[[0.1, 0.2], [-0.2, 0.3], [0.05, -0.1]]], dtype=np.float64)  # 一条序列、三个位置、每个向量两维。
    targets = np.array([0, 1, 2], dtype=np.int64)  # 三个位置各自的正确候选编号。
    states = sequence_hidden(vectors, previous_hidden, recurrent_weight, hidden_bias)  # 得到形状 (1,3,2) 的全部状态。
    direct = np.zeros_like(states)  # 保存各输出层直接传给对应状态的梯度。
    for time in range(states.shape[1]):  # 分别处理三个位置的输出层。
        scores = states[:, time, :] @ weight + bias  # 将本步状态转换为候选分数。
        _, _, score_grad = loss_and_grad(scores[0], int(targets[time]))  # 计算当前位置的分数梯度。
        _, _, hidden_grad = output_backward(states[:, time, :], weight, score_grad.reshape(1, -1) / states.shape[1])  # 平均损失的缩放只在这里处理一次。
        direct[:, time, :] = hidden_grad  # 这里只保存直接路径，未来路径由反向循环合并。
    grad_vectors, grad_initial, grad_recurrent, grad_hidden_bias = sequence_backward(states, previous_hidden, recurrent_weight, direct)  # 一次反向循环处理全部位置。
    numeric_vectors = numerical_array_gradient(vectors, lambda candidate: sequence_loss(candidate, previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets))  # 核对全部输入位置的梯度。
    numeric_recurrent = numerical_array_gradient(recurrent_weight, lambda candidate: sequence_loss(vectors, previous_hidden, candidate, hidden_bias, weight, bias, targets))  # 核对所有时间步累计的循环权重梯度。
    assert np.allclose(grad_vectors, numeric_vectors, rtol=1e-6, atol=1e-8), "序列输入梯度不正确"  # 检查所有输入分量。
    assert np.allclose(grad_recurrent, numeric_recurrent, rtol=1e-6, atol=1e-8), "序列循环权重梯度不正确"  # 检查共享权重的全部分量。
    print("序列状态形状：", states.shape)  # 应为 (1,3,2)。
    print("序列平均损失：", sequence_loss(vectors, previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets))  # 显示三处预测的平均损失。
    print("序列输入梯度：", grad_vectors)  # 每个位置都有自己的输入向量梯度。
    print("初始状态梯度：", grad_initial)  # 显示整段序列传回初始状态的梯度。
    print("序列循环权重梯度：", grad_recurrent)  # 显示三个时间步累计的参数梯度。
    print("序列输入最大绝对误差：", np.max(np.abs(grad_vectors - numeric_vectors)))  # 显示输入梯度核对误差。
    print("序列权重最大绝对误差：", np.max(np.abs(grad_recurrent - numeric_recurrent)))  # 显示循环权重核对误差。
    print("反向时间循环与数值检查：通过")  # 两项断言通过后显示。
```

### 老师预检与效率

老师组合上述原样代码在内存中运行，旧检查及新断言全部通过；另以固定上游线性目标独立检查 (B,T)=(1,1)、(1,2)、(3,4)，对输入向量、初始状态、共享循环权重、隐藏偏移四组数组做逐元素中心差分，全部通过，输入数组未修改。这些是老师预检，不是学习者完成记录。

本课不增加模型参数。循环层反向计算量约 O(B*T*H²)，保存状态和输入梯度各需 O(B*T*H) 空间，future 只需 O(B*H)。以 B=16、T=32、H=64、float32 为例，一个状态数组含 32768 个数，占 131072 字节，即 128 KiB，不包含分数、概率和其他数组。当前差分演示用 float64，单个同形状数组会占两倍空间。数值差分只用于核对，正式训练使用解析反向。

老师三步预检输出：

```text
序列状态形状： (1, 3, 2)
序列平均损失： 0.940437849047357
序列输入梯度： [[[-0.27062855  0.11539733]
  [ 0.11788239 -0.22584693]
  [ 0.29571928  0.07544122]]]
初始状态梯度： [[-0.11223481  0.1273475 ]]
序列循环权重梯度： [[-0.07040922 -0.03891982]
 [ 0.16874352 -0.02590774]]
序列输入最大绝对误差： 9.320988425542964e-12
序列权重最大绝对误差： 9.293871228166495e-12
反向时间循环与数值检查：通过
```

运行 python scripts/learn/09_softmax_gradient.py，提交新增输出并回答：一、T=4 时反向循环 time 的顺序是什么？二、为什么 future 每步用新值替换，而 grad_weight 要用 += 累加？学习者完成后再检查脚本、记录结果并完整同步 GitHub。

### 第十九课实操复查：通过，循环含义继续讲解

老师检查学习者实际脚本及 diff，并重新运行。全部旧断言和新增断言通过；三步平均损失 0.940437849047357，状态形状 (1,3,2)，输入梯度及共享循环权重梯度均与预期一致；最大差分误差分别 9.320988425542964e-12、9.293871228166495e-12。未修改学习者实现。已有 CI 已运行本脚本，不另增加重复检查。

学习者对两道口述题回答“我不懂”，因此记录为实操通过、概念待理解，继续本课，不引入新代码。

第一题拆开看：T=4 表示四个位置，range(4) 产生下标 0、1、2、3，reversed(range(4)) 则按 3、2、1、0 遍历。不是从 4 开始。先处理后一步，是因为前一步总梯度需要用到后一步传回的贡献。

第二题区分梯度的对象：future 在每次循环进入时，是更晚的步骤传给当前状态的梯度；通过本步后，新 future 变成传给前一个状态的梯度。新旧 future 针对不同状态，旧值已进入 total，不能额外加到新值上。grad_weight 始终针对同一组共享权重，逐步收集其每次使用的贡献，因此需要 +=。

为避免矩阵干扰，用一个单独的标量演示：h_t=x_t+0.5*h_(t-1)，目标 J=h1+h2+h3，三个位置的直接梯度均为 1。此处刻意省去 tanh 和交叉熵，只看梯度如何沿时间传递，不替换实际脚本的模型或损失。反传第三步时 future=0、total=1，传给 h2 的新 future=1*0.5=0.5；反传第二步时 total=1+0.5=1.5，传给 h1 的新 future=1.5*0.5=0.75。应把 future 替换为 0.75，而非累加成 1.25，因为旧的 0.5 是对 h2 的梯度，已经乘过局部导数传向 h1。

共享权重的分步贡献则始终对应同一个参数。举例某一权重元素在三个时间步的贡献（已包含目标缩放）为 0.2、-0.1、0.4，总梯度为 0.5；覆盖旧值只留下最后一份会漏掉其他贡献。真实循环层每步先计算 A_t=total*(1-h_t²)，新 future=A_t@W_h.T，而 step_weight=previous.T@A_t，随后累加到 grad_weight。

本次仅确认一个小问题：T=4 时，反向循环第一次取到的 time 是 3 还是 4？待学习者回答后继续确认梯度传递，不提前记为概念掌握。

### 反向下标复核：回答正确

学习者回答“取到3”，正确。T=4 时有效下标为 0、1、2、3，反向顺序为 3→2→1→0，第一次取 3。此处仅确认反向起始下标，future 替换与共享参数梯度累加的区别仍待复核。

下一步沿用前面的标量例子：进入第二步时旧 future=0.5，直接梯度为 1，总梯度为 1.5，乘循环连接导数 0.5 后得到传给前态的新梯度 0.75。请判断更新后的 future 应为 0.75，还是旧值加新值后的 1.25，并说明它现在传给 h1 还是 h2。保持问题围绕已讲内容，不引入新代码。

### future 更新纠错：不能将不同状态的梯度直接累加

学习者选择“加上去变成1.25累计总梯度”，此处不正确，说明仍将状态梯度传递与共享参数梯度累加混淆。本例旧 future=0.5 是对 h2 的后续贡献；它已经进入 total_h2=1+0.5=1.5，再乘 dh2/dh1=0.5，得到传给 h1 的新 future=0.75。新旧数值对应不同求导对象，不能直接相加。

把传给 h1 的 0.75 展开可见：1*0.5+0.5*0.5=0.5+0.25。第一项来自 h2 的直接目标项，第二项来自 h3 的目标项经两段循环连接传回。后续两处贡献已在新值中合并；再加旧 0.5 会额外加入一份尚未按当前连接转换的梯度。到第一步才把 h1 自身的直接梯度 1 加进去，得到 total_h1=1+0.75=1.75，然后继续传给初始状态。

规则应理解为“对同一个变量的各条路径贡献相加”；不是所有数值都相加，也不是状态梯度永远不能累加。共享参数 grad_weight 的各步贡献始终对同一个 W 求导，所以用 +=；future 在各次循环中对应不同状态，所以用本步反传得到的新值替换。

下一次只核对旧值是否已经参与计算：在 total_h2=1+0.5=1.5 中，后面传来的 0.5 是否已经被包含？此概念仍待掌握，不推进下一课。

### future 重复累加纠错后的确认

学习者回答“是的，理解了”，确认旧 future=0.5 已包含在 total_h2=1.5 中。正确结论是继续通过本步求导得到传给 h1 的新 future=0.75，并替换旧值，不再额外加旧 0.5。共享权重梯度则对同一个参数汇总各步贡献，仍使用 +=。

当前进度：第十九课代码与数值检查通过，反向循环起始下标回答正确，future 重复累加误解经讲解后已获学习者确认；后续实践中继续巩固，不据此声称已经独立掌握全部 BPTT 推导。本次不新增练习代码。

## 第二十课：字符表示表反向传播与重复编号累加

状态：老师已在内存中预检以下代码；待学习者在 scripts/learn/09_softmax_gradient.py 手敲、运行和回答。未替学习者修改脚本，不代表本课已经完成。

### 从位置梯度回到参数梯度

查表前向 vectors[b,t,:]=E[token_ids[b,t],:]。E 是形状 (V,H) 的可训练参数表，token_ids 是 (B,T) 整数编号，vectors 和 grad_vectors 是 (B,T,H)。编号只用于选行，本课不对离散整数编号求导；需要更新的是选中行的浮点参数。

固定编号 i、向量分量 j，∂L/∂E[i,j]=Σ_{b,t:token_ids[b,t]=i} ∂L/∂vectors[b,t,j]。选中行对当前位置向量是直接复制，局部导数为 1；其他行的局部导数为 0。同一编号可以在同一序列或不同序列出现多次，这些位置共用同一行参数，所以贡献相加。grad_vectors 已经包含跨时间反传及平均损失缩放，表示表汇总时不再除以出现次数或批次位置数。

例如输入编号 [[0,1,0]]，三个位置的向量梯度依次为 [0.2,-0.1]、[0.4,0.3]、[-0.5,0.6]，那么第零行梯度为 [-0.3,0.5]，第一行为 [0.4,0.3]，未使用行全零。同一字符的向量相同，但各使用位置的向量梯度可以因上下文、标签及后续路径不同而不同。

### 第一块：main 前添加表示表反向函数

```python
def embedding_backward(token_ids, embedding, grad_vectors):  # 把各位置的向量梯度汇总到表示表对应行。
    assert token_ids.ndim == 2 and embedding.ndim == 2, "需要二维编号和二维表示表"  # 编号是 (B,T)，表示表是 (V,H)。
    assert np.issubdtype(token_ids.dtype, np.integer), "字符编号必须为整数"  # 编号用于查找表示表的行。
    assert ((token_ids >= 0) & (token_ids < embedding.shape[0])).all(), "字符编号越界"  # 拒绝负数或超过词表范围的编号。
    assert grad_vectors.shape == token_ids.shape + (embedding.shape[1],), "向量梯度形状不对"  # 每个输入位置必须有 H 维梯度。
    grad_embedding = np.zeros_like(embedding)  # 为每个表示表参数准备一个梯度，初始为零。
    for batch in range(token_ids.shape[0]):  # 逐条处理批次中的序列。
        for time in range(token_ids.shape[1]):  # 逐个处理当前序列的位置。
            token = token_ids[batch, time]  # 找到当前位置使用的是表示表哪一行。
            grad_embedding[token] += grad_vectors[batch, time]  # 将本位置贡献加到对应行，重复编号也会正确累加。
    return grad_embedding  # 返回形状 (V,H) 的表示表总梯度，不修改原参数。
```

这里用两层循环逐位置写 +=，方便看清重复编号的每份贡献。不要把循环体改成直接赋值，否则重复字符前面的贡献会被覆盖。也不要直接用批量高级索引 grad_embedding[token_ids] += grad_vectors 来替代循环，重复索引时不保证按每次出现正确累加。后面优化时再讨论专门的索引累加操作。这里的浮点表示表沿用模型参数的浮点类型。

### 第二块：在原 main 末尾添加手算检查

```python
    small_embedding = np.array([[0.1, 0.2], [-0.2, 0.3], [0.05, -0.1], [0.2, -0.2]], dtype=np.float64)  # 建立四行两列的小表示表。
    token_ids = np.array([[0, 1, 0]], dtype=np.int64)  # 编号 0 使用两次，编号 1 使用一次。
    toy_vector_grad = np.array([[[0.2, -0.1], [0.4, 0.3], [-0.5, 0.6]]], dtype=np.float64)  # 手动给定三个位置的梯度，先单独检查汇总规则。
    toy_table_grad = embedding_backward(token_ids, small_embedding, toy_vector_grad)  # 将位置梯度汇总成表示表梯度。
    expected_table_grad = np.array([[-0.3, 0.5], [0.4, 0.3], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)  # 第零行收到两次贡献，未使用行保持零。
    assert np.allclose(toy_table_grad, expected_table_grad), "重复编号汇总不正确"  # 确认没有覆盖同一字符较早的贡献。
    print("手算表示表梯度：", toy_table_grad)  # 显示每个编号对应行的梯度。
```

这段梯度是人为给定，只用于验证按编号汇总，没有经过 RNN，不能当作真实训练梯度。下一块再接真实损失。

### 第三块：紧接着添加完整链路检查

保留旧变量 previous_hidden、recurrent_weight、hidden_bias、weight、bias 和上一课 targets=[0,1,2]；以下仍在 main 内，整体缩进四格。表大小 V=4，输出候选数为 3，这是只为复用旧玩具输出层的梯度实验，输入输出维数允许不同；最终真实语言模型输入输出都用同一个词表大小 5547。

```python
    table_vectors = small_embedding[token_ids]  # 按编号查表，得到形状 (1,3,2) 的三个位置向量。
    table_states = sequence_hidden(table_vectors, previous_hidden, recurrent_weight, hidden_bias)  # 将查表结果送入整段循环计算。
    table_direct = np.zeros_like(table_states)  # 准备各位置输出层直接传回的状态梯度。
    for time in range(table_states.shape[1]):  # 逐位置计算输出层反向传播。
        table_scores = table_states[:, time, :] @ weight + bias  # 使用原来的输出权重和偏移计算分数。
        _, _, table_score_grad = loss_and_grad(table_scores[0], int(targets[time]))  # 沿用上一课的三个标签 0、1、2。
        _, _, table_hidden_grad = output_backward(table_states[:, time, :], weight, table_score_grad.reshape(1, -1) / table_states.shape[1])  # 平均损失的缩放只处理一次。
        table_direct[:, time, :] = table_hidden_grad  # 保存输出层传给本状态的直接贡献。
    table_vector_grad, _, _, _ = sequence_backward(table_states, previous_hidden, recurrent_weight, table_direct)  # 经过完整反向循环，得到各输入位置的梯度。
    table_grad = embedding_backward(token_ids, small_embedding, table_vector_grad)  # 最后通过查表，把梯度送回共享表示表。
    numeric_table = numerical_array_gradient(small_embedding, lambda candidate: sequence_loss(candidate[token_ids], previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets))  # 每次改变一个表元素后重新查表并重算完整损失。
    assert np.allclose(table_grad, numeric_table, rtol=1e-6, atol=1e-8), "表示表梯度不正确"  # 逐元素核对整张表，包括未使用的行。
    assert np.allclose(table_grad[0], table_vector_grad[0, 0] + table_vector_grad[0, 2]), "重复字符贡献没有合并"  # 第零行要收到两个使用位置的梯度。
    assert np.array_equal(table_grad[2:], np.zeros_like(table_grad[2:])), "未使用行的梯度应为零"  # 本例输出层权重独立，表示表只通过输入查表参与计算。
    table_before = sequence_loss(small_embedding[token_ids], previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets)  # 记录更新表示表之前的损失。
    updated_embedding = small_embedding - 0.1 * table_grad  # 仅更新演示表示表，其他参数保持固定。
    table_after = sequence_loss(updated_embedding[token_ids], previous_hidden, recurrent_weight, hidden_bias, weight, bias, targets)  # 重新查更新后的表并计算损失。
    assert table_after < table_before, "本例表示表更新没有降低损失"  # 核对这个具体学习率下的单步结果。
    print("真实表示表梯度：", table_grad)  # 显示通过损失反传得到的梯度。
    print("表示表梯度最大绝对误差：", np.max(np.abs(table_grad - numeric_table)))  # 显示解析梯度与数值差分的误差。
    print("仅更新表示表前后损失：", table_before, table_after)  # 观察向量参数更新如何影响预测损失。
    print("表示表反向、重复编号与单步更新检查：通过")  # 所有断言通过后显示。
```

这次第一个和第三个输入都取表示表第零行。它们的输入向量相同，状态和梯度可能不同。不要复用上一课原 vectors 的梯度，因为旧第三向量与新查表结果不同；本块重新前向与反向。数值差分必须使用 candidate[token_ids] 重新查表，同一候选行的变动会同时影响它的所有使用位置。

本模型输入表示表 E 与输出权重 W_o 是独立参数，没有权重绑定。未作为输入出现的表示表行梯度为零；即使该编号作为输出候选出现，也不改变此结论。这里检查的是原始损失梯度，尚未加权重衰减等其他更新规则。正确性检查后仅对演示表示表更新一次，以观察损失变化，不保证任意学习率都下降，也尚未整合完整模型训练。

### 参数、效率与老师预检

这一步不增加参数，只补齐已有表示表的梯度。真实词表 V=5547、H=64，共 V*H=355008 个表示表参数；float32 下参数表占 1420032 字节，稠密梯度表另占相同大小。完整模型参数仍为 719723。初始化稠密梯度表需要 O(VH) 时间与空间，按位置累加需要 O(BTH) 时间；不需要构造形状 (B,T,V) 的独热数组。此处未比较运行耗时，Python 循环实现后续可以优化。

老师在内存中将本课代码与现有脚本组合运行，全部旧检查和新断言通过；另用两条序列、跨序列重复编号的固定线性目标检查整张表示表梯度，核对输入编号、参数表和上游梯度未被改写。这些为老师预检，学习者结果待提交。

老师参考输出：

```text
手算表示表梯度： [[-0.3  0.5]
 [ 0.4  0.3]
 [ 0.   0. ]
 [ 0.   0. ]]
真实表示表梯度： [[ 0.03322404  0.21615905]
 [ 0.12444289 -0.21856423]
 [ 0.          0.        ]
 [ 0.          0.        ]]
表示表梯度最大绝对误差： 1.2712025876382427e-11
仅更新表示表前后损失： 0.9821935073921791 0.9711489072847838
表示表反向、重复编号与单步更新检查：通过
```

运行 python scripts/learn/09_softmax_gradient.py，提交新增输出。口述题：一、同一编号两次出现的梯度分别为 [0.2,-0.1] 和 [-0.5,0.6]，对应表示表行的总梯度是多少？二、如果写 grad_embedding[token] = grad_vectors[batch,time] 而不是 +=，重复编号时会发生什么？完成后复查实际脚本、补充笔记并上传 GitHub。

### 第二十课实操复查：通过，重复编号累加继续讲解

老师检查学习者实际代码及 diff，重新运行脚本，全部旧检查与新增检查通过，未修改学习者代码。手算梯度符合预期；真实表示表第零行为 [0.03322404,0.21615905]，第一行为 [0.12444289,-0.21856423]，未使用行均为零；差分最大绝对误差 1.2712025876382427e-11；仅更新表示表前后损失从 0.9821935073921791 降至 0.9711489072847838。已有 CI 会执行此脚本，不新增重复任务。

两道口述题学习者回答“我不知道”。因此仅记录实操通过，概念继续讲解，不推进下一课。

第一题逐个分量相加：[0.2+(-0.5),-0.1+0.6]=[-0.3,0.5]。两个位置虽然不同，但编号都是 0，共用同一参数行，所以把各位置对这行参数的影响合起来，不是选择其中一个，也不再求平均。该汇总得到的是梯度，尚未更新表示表参数；后面才按参数减去学习率乘梯度更新。

第二题逐次观察第零行的梯度缓冲区。起始为 [0,0]；第一次遇到编号 0 时，无论用 = 还是 += 都得到 [0.2,-0.1]；中间遇到编号 1 只改第一行，不影响第零行；再次遇到编号 0 时，用 += 得到 [-0.3,0.5]，而用 = 会覆盖成 [-0.5,0.6]，漏掉第一次出现的贡献。注意这里改的是 grad_embedding 梯度缓冲区，不是直接修改 embedding 参数表。

本次只确认一个更小的问题：若使用 =，第二次处理编号 0 后，第零行留下的是第一次贡献 [0.2,-0.1]，还是第二次贡献 [-0.5,0.6]？待学习者回答，不提前标记概念已掌握。

### 赋值覆盖复核：回答正确

学习者回答“第二次”，正确：使用 = 时，第二次处理编号 0 会将第零行梯度覆盖为 [-0.5,0.6]，第一次的 [0.2,-0.1] 被丢弃。使用 += 才会保留两次贡献，得到 [-0.3,0.5]。本次确认的是赋值覆盖的区别，不据此声称已独立掌握全部表示表反向推导。

当前进度：第二十课代码与数值检查通过，赋值覆盖问题回答正确；继续用一个简单加法巩固梯度汇总：同一行已有梯度 0.2，本次贡献为 -0.5，使用 += 后该分量是多少？本次不新增练习代码。

### 梯度累加算术复核：通过

学习者回答“-03”，结合题意按漏写小数点的 -0.3 理解，结果正确：0.2+(-0.5)=-0.3。规范写法为 -0.3。第二十课实操和数值检查通过，学习者已正确辨认 = 会覆盖前一次贡献，并正确计算 += 的一个分量；后续整合模型时继续巩固按字符编号累加，不据此声称已独立掌握全部求导过程。本次未修改练习脚本。

下一课计划：把已经验证的输出层、时间反向循环和表示表反向整合到实际 TinyRNN，返回与五组参数一一对应的梯度；再做小规模整体梯度检查和一次统一参数更新。该整合尚未实施。

## 第二十一课：将完整反向传播整合进 TinyRNN

状态：老师已在内存中组合预检以下代码，待学习者在 scripts/learn/07_tiny_rnn.py 手敲、运行并回答。scripts/learn/09_softmax_gradient.py 保留作推导实验，不继续堆新代码；未替学习者修改实际模型。本课不进行语料训练，只在短例子上求完整梯度并验证统一单步更新。

### 五组参数与整体依赖

embedding 为 (V,H)，recurrent 为 (H,H)，bias 为 (H,)，output 为 (H,V)，output_bias 为 (V,)，每组梯度与对应参数同形状。参数总数仍为 2VH+H²+H+V=719723，没有增加参数。loss_and_grad 返回 (loss,grads)，grads 的键与 self.params 一致，计算过程中不修改任何参数。

前向：字符编号→查表示表→全部状态→输出分数→稳定平均交叉熵。反向：分数梯度 (p-q)/(B*T)→输出权重/偏移梯度与各状态直接梯度→反向时间循环→循环权重/隐藏偏移累加与表示表行累加。用已有 forward_hidden 计算一次状态，在同一方法中复用状态计算输出，不重复调用 forward 丢失缓存。

### 第一块：仅修改 forward_hidden 中两处 dtype

现有前向把隐藏数组固定为 float32。为了能在小模型中使用双精度差分，把下面两行的 dtype 改为 vectors.dtype，其余逻辑保留。正常参数仍初始化为 float32，所以正式模型类型不变。

```python
        hidden = np.zeros((batch_size, self.hidden_size), dtype=vectors.dtype)  # 初始状态类型跟随查表向量，支持双精度检查。
        states = np.zeros((batch_size, seq_length, self.hidden_size), dtype=vectors.dtype)  # 保存状态时保留相同精度，避免检查模型被截回单精度。
```

### 第二块：在 TinyRNN 类内添加方法的前半段

放在 forward 方法后、顶格 def main() 前。方法定义缩进四个空格，方法体缩进八格。下一块直接续写在同一个方法内部。

```python
    def loss_and_grad(self, token_ids, targets):  # 计算平均损失和五组参数梯度，暂时不更新参数。
        token_ids = np.asarray(token_ids)  # 将输入整理成数组，合法性由已有前向检查。
        targets = np.asarray(targets)  # 保留标签原来的整数类型，交叉熵会检查范围与形状。
        states = self.forward_hidden(token_ids)  # 保存全部时间位置的状态，供输出层和循环反向使用。
        logits = states @ self.params["output"] + self.params["output_bias"]  # 使用同一次前向状态计算分数。
        loss, _ = cross_entropy(logits, targets)  # 使用已有稳定交叉熵，得到所有位置的平均损失。
        batch_size, seq_length, _ = states.shape  # 读取实际批大小和窗口长度。
        grad_logits = softmax(logits)  # 分数梯度先从每个位置的预测概率开始。
        flat_grad = grad_logits.reshape(-1, self.vocab_size)  # 合并批次和时间轴，方便按正确编号减一。
        flat_grad[np.arange(targets.size), targets.reshape(-1)] -= 1.0  # 每个位置的正确类别减一，实现 p-q。
        grad_logits /= targets.size  # 平均损失除以实际位置总数 B*T，只在这里缩放一次。
        grads = {name: np.zeros_like(param) for name, param in self.params.items()}  # 为五组参数准备同形状同类型的梯度。
        flat_states = states.reshape(-1, self.hidden_size)  # 将全部位置的状态整理成 (B*T,H)。
        grads["output"] = flat_states.T @ flat_grad  # 汇总所有位置对共享输出权重的贡献。
        grads["output_bias"] = flat_grad.sum(axis=0)  # 汇总所有位置对输出偏移的贡献。
        direct = grad_logits @ self.params["output"].T  # 输出层直接传回各状态的梯度，形状为 (B,T,H)。
```

flat_grad 是 grad_logits 的 reshape 视图；softmax 返回的数组可按这个顺序共享底层数据，因此按正确编号减一后，原 grad_logits 也同步改变，原位 /= 实际位置数后 flat_grad 也获得缩放结果。不要将 flat_grad 改成 .copy() 后仍假定二者同步；本方法使用的 reshape 对当前连续数组为视图。合并 B、T 时每个预测位置有唯一行号，因此这里减一的索引没有重复行，不涉及表示表重复编号累加问题。

输出层计算：N=B*T，flat_states 为 (N,H)，flat_grad 为 (N,V)，所以 dW_o=flat_states.T@flat_grad 为 (H,V)，db_o=flat_grad.sum(axis=0) 为 (V,)，direct 为 (B,T,H)。矩阵乘法和求和已汇总全部位置，后面不再重复累加输出梯度。

### 第三块：续写同一个方法的循环反向部分

```python
        initial = np.zeros((batch_size, self.hidden_size), dtype=states.dtype)  # 本模型每个窗口都从全零状态开始。
        future = np.zeros_like(initial)  # 最后一个位置之后没有更多时间步，后续贡献从零开始。
        for time in reversed(range(seq_length)):  # 从最后一个位置向第零个位置反传。
            previous = initial if time == 0 else states[:, time - 1, :]  # 取本步前态，第零步取初始状态。
            total = direct[:, time, :] + future  # 合并当前输出与后续步骤传来的梯度。
            activation_grad = total * (1.0 - states[:, time, :] ** 2)  # 通过本步 tanh，得到激活前梯度。
            grads["recurrent"] += previous.T @ activation_grad  # 累加本步对共享循环权重的贡献。
            grads["bias"] += activation_grad.sum(axis=0)  # 累加本步对隐藏偏移的贡献。
            for batch in range(batch_size):  # 遍历本时间步各条序列的字符编号。
                grads["embedding"][token_ids[batch, time]] += activation_grad[batch]  # 将输入向量梯度累加到对应字符行。
            future = activation_grad @ self.params["recurrent"].T  # 用本步算出的梯度替换 future，继续传给前态。
        return loss, grads  # 返回损失和五组完整梯度，调用者之后再统一更新参数。
```

本模型输入向量直接相加，因此 activation_grad 也是对应输入向量的梯度，可以马上按编号累加到表示表，不必再单独存一个 (B,T,H) 的输入梯度数组。future 每步替换，共享参数梯度每步累加；所有参数保持本次前向使用的原值，完整反向结束后才更新。

### 第四块：在原 main 末尾添加整体检查与统一更新

保留原 main 中 model、toy_x、toy_y、loss 等变量，以下整体缩进四格，放在原 main 的最后一个 print 后、文件末尾 if __name__ 判断前。新建 V=4、H=2 的检查模型，参数数为 8+4+2+8+4=26。双精度只用于此小模型检查，不检查真实大模型的每个参数。

```python
    full_loss, full_grads = model.loss_and_grad(toy_x, toy_y)  # 在真实词表大小的模型上计算完整梯度。
    assert np.isclose(full_loss, loss), "整合后的损失与原前向不一致"  # 对同一输入和标签，损失定义应保持一致。
    for name, param in model.params.items():  # 检查真实模型五组梯度的形状、类型和数值。
        assert full_grads[name].shape == param.shape and full_grads[name].dtype == param.dtype and np.isfinite(full_grads[name]).all(), name + "梯度异常"  # 每个参数都应有同形状的有限梯度。
    print("完整模型梯度形状：", {name: grad.shape for name, grad in full_grads.items()})  # 显示实际五组梯度的形状。
    check_model = TinyRNN(vocab_size=4, hidden_size=2, seed=42)  # 创建只有 26 个参数的小模型，避免逐个差分检查大模型。
    check_model.params = {name: param.astype(np.float64) for name, param in check_model.params.items()}  # 只把检查模型改为双精度，正式模型仍用 float32。
    check_x = np.array([[0, 1, 0], [1, 2, 1]], dtype=np.int64)  # 两条序列含重复字符，也包含跨序列重复编号。
    check_y = np.array([[1, 0, 2], [2, 1, 0]], dtype=np.int64)  # 每个输入位置指定一个下一字目标编号。
    saved_params = {name: param.copy() for name, param in check_model.params.items()}  # 保存参数，检查计算梯度不会偷偷更新模型。
    before, check_grads = check_model.loss_and_grad(check_x, check_y)  # 一次解析反向得到全部参数梯度。
    assert all(np.array_equal(param, saved_params[name]) for name, param in check_model.params.items()), "求梯度时修改了参数"  # 更新必须发生在完整反向结束之后。
    epsilon = 1e-5  # 双精度中心差分使用的小试探距离。
    for name, param in check_model.params.items():  # 逐组核对五种参数的梯度。
        numeric = np.zeros_like(param)  # 为当前参数组准备数值梯度。
        for index in np.ndindex(param.shape):  # 逐个检查当前参数组中的标量。
            original = float(param[index])  # 记住本元素原值，试探完必须恢复。
            param[index] = original + epsilon  # 向正方向试探当前元素。
            plus, _ = cross_entropy(check_model.forward(check_x), check_y)  # 重新前向，得到正向试探损失。
            param[index] = original - epsilon  # 从原值向负方向试探同一元素。
            minus, _ = cross_entropy(check_model.forward(check_x), check_y)  # 重新前向，得到负向试探损失。
            param[index] = original  # 恢复原参数，避免影响其他元素的检查。
            numeric[index] = (plus - minus) / (2.0 * epsilon)  # 用中心差分估计该元素的偏导数。
        assert np.allclose(check_grads[name], numeric, rtol=1e-5, atol=1e-8), name + "整体梯度检查失败"  # 对比当前组的所有解析与数值梯度。
        print(name + "整体梯度最大误差：", np.max(np.abs(check_grads[name] - numeric)))  # 输出当前参数组的最大绝对误差。
    assert np.array_equal(check_grads["embedding"][3], np.zeros(2)), "未使用的输入表示行梯度应为零"  # 输出权重与表示表独立，未作为输入出现的行不受损失影响。
    for name in check_model.params:  # 使用同一次完整反向得到的梯度，统一更新全部参数组。
        check_model.params[name] -= 0.1 * check_grads[name]  # 每个参数沿负梯度方向走一个小步。
    after, _ = cross_entropy(check_model.forward(check_x), check_y)  # 所有参数更新完后再重新计算损失。
    assert after < before, "本例完整参数更新没有降低损失"  # 只验证本例与本学习率下的结果。
    print("完整模型单步更新前后损失：", before, after)  # 显示完整模型一次更新带来的变化。
    print("完整梯度与统一单步更新检查：通过")  # 五组差分和单步更新都通过后显示。
```

数值差分改变一个参数元素后，调用正常前向与交叉熵重新计算损失，再恢复参数；不调用解析反向当作数值参考。五组梯度都基于更新前的同一份参数，更新循环中不重新计算梯度，全部更新完成后再前向。本例学习率 0.1 能降低损失，不是任意学习率都必然下降的保证。

### 计算量、内存与老师预检

真实模型 float32 参数占 719723*4=2878892 字节；五组稠密梯度再占相同大小，不含状态、分数、概率等中间数组，仍远低于 1 GB。完整反向主要计算量约 O(B*T*(H²+H*V))，另有 O(参数数量) 的梯度数组初始化。若逐标量差分，每个标量至少需要两次前向；大模型一次全量核对约需 1439446 次前向，而 26 参数检查只需 52 次，因此先用极小模型验证算法。

老师将以上修改应用于内存中的源码，运行原 main 与本课全部检查通过。另检查 (B,T)=(1,1)、(1,4)、(3,2) 的全部五组参数中心差分，检查计算梯度不改参数，并确认等长多序列批次的平均梯度等于逐行平均梯度的均值。以下为老师预检输出，待学习者实际完成后另行记录。

```text
完整模型梯度形状： {'embedding': (5547, 64), 'recurrent': (64, 64), 'bias': (64,), 'output': (64, 5547), 'output_bias': (5547,)}
embedding整体梯度最大误差： 2.73799455330348e-11
recurrent整体梯度最大误差： 1.0723519294764117e-11
bias整体梯度最大误差： 3.5172004198003037e-12
output整体梯度最大误差： 2.6684005699495472e-11
output_bias整体梯度最大误差： 1.301336816084131e-11
完整模型单步更新前后损失： 1.3645426721710658 1.3442277271788379
完整梯度与统一单步更新检查：通过
```

运行 python scripts/learn/07_tiny_rnn.py，原有前向与交叉熵检查也必须通过。提交新增输出，并回答：一、批大小 B=2、窗口长度 T=3 时，分数梯度应该除以多少？二、为什么不能在反向时间循环里一边求梯度一边马上修改 recurrent 权重？学习者完成后复查代码、记录实际结果、更新 CI 说明并同步 GitHub。

### 第二十一课实操复查：通过，更新时机继续讲解

老师检查学习者实际脚本及 diff 并重新运行，全部原有检查及五组完整梯度检查通过。实际五组最大绝对误差依次为：embedding 2.73799455330348e-11、recurrent 1.0723519294764117e-11、bias 3.5172004198003037e-12、output 2.6684005699495472e-11、output_bias 1.301336816084131e-11；完整小模型更新前后损失从 1.3645426721710658 降至 1.3442277271788379。真实模型梯度形状与五组参数完全一致。

学习者保留了旧 float32 初始化行作为注释，新计算使用 vectors.dtype，逻辑正确，无需删除学习对照注释。老师仅清理行尾空白，清理前后语法树一致，未修改实现。CI 原已运行 07 和 09，本次更新其说明，实际检查仍由脚本断言执行。

第一题回答“应该除以6”，正确：B*T=2*3=6，目标是六个预测位置的平均交叉熵，所以分数梯度为 (p-q)/6，后续不再重复除。第二题回答“不知道”，记录为实操与第一题通过、更新时机概念待复核。

反向计算求的是本次前向所对应的损失函数，在同一份参数处的梯度。前向保存的状态来自旧权重，反向沿循环传递时又需要乘同一旧权重的转置。如果反传到一半就修改 recurrent，那么后续计算会混用旧状态与新权重；各部分不再属于同一次前向、同一个参数点的完整梯度，而且共享参数各时间步的贡献还没有收集完。

用标量例子复习：h1=w、h2=h1*w、L=h2，w=2 时，前向保存 h1=2，完整梯度应为 2+2=4。若取得最后一次乘法的直接贡献 2 后，就将共享 w 改成 1.9，再用新 w 向 h1 反传，另一条贡献会变成 1.9，混合结果为 3.9。它既不是旧点 w=2 的梯度 4，也不是重新前向后新点 w=1.9 的梯度 3.8。该例只说明更新中途混用参数的问题，不替代模型实际的 tanh 与交叉熵。

正确顺序：固定参数完成前向→固定同一份参数完成全部反向与梯度汇总→统一执行参数减学习率乘梯度→用新参数进行下一次前向。这里“统一更新”指使用同一参数点得到的五组梯度；代码可以按参数名逐组更新，但更新过程中不能重新反传或混入新计算的梯度。本课尚未开始完整语料训练。

待确认的小问题：本次前向用 w=2，反向还没结束时，继续传递梯度应使用原来的 2，还是先改成新值 1.9？等待学习者回答，不提前认定概念已掌握。

### 参数更新时机复核：回答正确

学习者回答“反向没结束时应该还是2”，正确。本次前向使用 w=2，完整反向也必须保持这份参数不变；全部时间步的贡献收集完，才用完整梯度更新参数。下一次前向与反向再使用更新后的参数。

第二十一课实际脚本和五组参数的整体差分检查通过，平均损失缩放题与参数保持不变题均回答正确。本次仅补记理解确认，未修改练习代码。下一阶段可学习梯度范数与梯度裁剪，为完整训练循环做准备；尚未实施新的课程代码或正式语料训练。

## 第二十二课：梯度全局范数与梯度裁剪

状态：老师已在内存中预检以下代码，待学习者在 scripts/learn/07_tiny_rnn.py 亲自输入、运行和回答。未替学习者修改脚本。本课用小模型演示，不启动语料训练，不宣称已经在本项目训练中观察到梯度爆炸。

### 梯度有方向，也有长度

把五组参数的所有梯度元素按任意固定顺序排成一个长向量 g，全局 L2 范数 ||g||₂=sqrt(Σ_i g_i²)。它是全部参数梯度的总长度，不是梯度元素之和、最大元素、平均值，也不是每组各算各的阈值。例如 [3,4] 的长度 sqrt(9+16)=5；[3,0,4] 长度同样为 5。

阈值 C>0，缩放比例 s=1（||g||≤C），或 s=C/||g||（||g||>C），最终裁剪梯度 g_clip=s*g。所有参数组使用同一个正比例，保留整体方向与各分量的相对比例。没有超过阈值时不改变，不把小梯度放大；零梯度保持零。

例如原梯度 [3,4]、C=2.5，s=0.5，裁剪后 [1.5,2]，长度为 2.5。逐元素直接限制在 [-2.5,2.5] 则会得到 [2.5,2.5]，不是本课的全局范数裁剪，方向也不同。

普通梯度下降 Δθ=-η*g_clip，所以 ||Δθ||₂=η*||g_clip||₂≤η*C。例如 η=0.1、C=0.1，更新长度最多 0.01。这一直接关系针对当前普通梯度下降；以后若使用动量或 Adam，需要按其更新公式重新分析。裁剪限制过大的更新，不保证每次损失都下降，也不能修复梯度消失。RNN 多步反向含连续矩阵乘法，梯度可能衰减或放大，裁剪是控制过大梯度的一种手段。

正确位置：完整前向→完整反向并汇总→对全部参数的平均损失梯度做一次全局裁剪→统一更新参数。不要在各时间步汇总完成前分别裁剪，也不要改写 loss_and_grad 来返回裁剪结果后再拿它与损失数值导数对比；数值检查验证的应是原始梯度。

### 第一块：在 TinyRNN 类之前顶格添加两个函数

```python
def gradient_norm(grads):  # 把全部参数梯度看作一个长向量，计算它的总长度。
    assert grads, "梯度字典不能为空"  # 至少需要一组梯度才能检查。
    squared_sum = 0.0  # 从零开始累加所有梯度分量的平方。
    for grad in grads.values():  # 遍历五组参数各自的梯度数组。
        assert np.issubdtype(grad.dtype, np.floating) and np.isfinite(grad).all(), "梯度必须是有限浮点数"  # 拒绝异常梯度，避免把错误带入更新。
        squared_sum += float(np.sum(grad.astype(np.float64) ** 2))  # 用双精度计算平方和，降低单精度累计误差。
    assert np.isfinite(squared_sum), "梯度平方和超出可计算范围"  # 平方和必须有限才能计算缩放比例。
    return float(np.sqrt(squared_sum))  # 对总平方和开根号，得到全局 L2 范数。
```

```python
def clip_gradients(grads, max_norm):  # 超过阈值时，将全部梯度按同一个比例缩小。
    assert np.isfinite(max_norm) and max_norm > 0, "裁剪阈值必须是有限正数"  # 阈值表示允许的最大梯度长度。
    norm = gradient_norm(grads)  # 在裁剪之前计算所有参数梯度的全局长度。
    scale = max_norm / norm if norm > max_norm else 1.0  # 超过阈值才缩小，零梯度也不会除以零。
    clipped = {name: grad * scale for name, grad in grads.items()}  # 返回新的梯度数组，保留原始梯度用于检查。
    return clipped, norm, scale  # 返回裁剪结果、原始长度和本次缩放比例。
```

浮点累加用 float64，而返回的裁剪数组保留现有 float32/float64 梯度类型。这里是面向正常有限训练梯度的简明平方和实现；不把 NaN/Inf 静默改为零，若平方和溢出会明确报错，不能用裁剪掩盖异常数值。条件分支已经处理零范数，不需要分母额外加 epsilon；本函数 max_norm 是阈值，和学习率、数值差分 epsilon 是不同概念。

### 第二块：在原 main 末尾添加手算及边界检查

整体缩进四个空格，保留上一课代码。

```python
    demo_grads = {"first": np.array([3.0, 0.0]), "second": np.array([[4.0]])}  # 用两组参数模拟合起来为 [3,0,4] 的梯度。
    demo_saved = {name: grad.copy() for name, grad in demo_grads.items()}  # 保存原始梯度，检查裁剪函数不改输入。
    demo_clipped, demo_norm, demo_scale = clip_gradients(demo_grads, max_norm=2.5)  # 总长度为 5，阈值为 2.5，所以按一半缩小。
    assert np.isclose(demo_norm, 5.0) and np.isclose(demo_scale, 0.5), "范数或比例计算不正确"  # 核对勾股定理和缩放公式。
    assert np.allclose(demo_clipped["first"], [1.5, 0.0]) and np.allclose(demo_clipped["second"], [[2.0]]), "各组没有按相同比例缩放"  # 所有组必须使用同一个系数。
    assert all(np.array_equal(grad, demo_saved[name]) for name, grad in demo_grads.items()), "裁剪修改了原始梯度"  # 原始梯度留作对照。
    unchanged, _, unchanged_scale = clip_gradients(demo_grads, max_norm=10.0)  # 阈值足够大时，梯度数值应保持不变。
    assert unchanged_scale == 1.0 and all(np.array_equal(unchanged[name], grad) for name, grad in demo_grads.items()), "未超阈值也被改变"  # 小梯度不能被放大到阈值。
    zero_clipped, zero_norm, zero_scale = clip_gradients({"zero": np.zeros(2)}, max_norm=1.0)  # 检查全部梯度为零的情况。
    assert zero_norm == 0.0 and zero_scale == 1.0 and np.array_equal(zero_clipped["zero"], np.zeros(2)), "零梯度处理错误"  # 零梯度原样保留。
    print("演示裁剪前范数：", demo_norm)  # 应为 5。
    print("演示缩放比例：", demo_scale)  # 应为 0.5。
    print("演示裁剪后范数：", gradient_norm(demo_clipped))  # 应为 2.5。
```

这里将 [3,0]、[[4]] 放在两个不同参数组中，检查它们合起来的全局范数，不能分别对两组使用阈值。原始梯度保留不变，裁剪函数返回独立数组；即使不需要裁剪，本实现也返回新数组。

### 第三块：紧接着添加小模型裁剪更新

继续沿用上一课的 check_model、check_x、check_y。注意 check_model 已经做过上一课的一次更新，所以必须重新调用 loss_and_grad，不能使用旧 check_grads。本例的 0.1 阈值只为清楚观察缩放，不是对正式语料训练选好的超参数。

```python
    clip_loss_before, raw_grads = check_model.loss_and_grad(check_x, check_y)  # 用上一课已更新过的小模型，重新完成一次前向和反向。
    clipped_grads, raw_norm, clip_scale = clip_gradients(raw_grads, max_norm=0.1)  # 对五组完整梯度进行全局裁剪。
    clipped_norm = gradient_norm(clipped_grads)  # 检查缩放后梯度的实际总长度。
    assert clipped_norm <= 0.1 + 1e-12, "裁剪后仍超过阈值"  # 为双精度舍入误差保留很小容差。
    params_before_clip_step = {name: param.copy() for name, param in check_model.params.items()}  # 保存更新前参数，用来计算实际更新长度。
    clip_learning_rate = 0.1  # 为本次小模型演示指定学习率。
    for name in check_model.params:  # 完成全部裁剪后，再逐组更新参数。
        check_model.params[name] -= clip_learning_rate * clipped_grads[name]  # 使用裁剪梯度执行一次普通梯度下降。
    parameter_change = {name: param - params_before_clip_step[name] for name, param in check_model.params.items()}  # 收集各参数的实际变化。
    update_norm = gradient_norm(parameter_change)  # 计算全部参数合起来移动了多远。
    assert np.isclose(update_norm, clip_learning_rate * clipped_norm, rtol=1e-8, atol=1e-12), "普通梯度下降更新长度不匹配"  # 更新长度应为学习率乘裁剪后梯度长度。
    clip_loss_after, _ = cross_entropy(check_model.forward(check_x), check_y)  # 用更新后参数重新计算损失。
    assert clip_loss_after < clip_loss_before, "本例裁剪后单步更新没有降低损失"  # 仅验证本例和当前学习率下的结果。
    print("小模型原始梯度范数：", raw_norm)  # 显示本批完整原始梯度有多大。
    print("小模型裁剪缩放比例：", clip_scale)  # 显示五组梯度共同使用的系数。
    print("小模型裁剪后梯度范数：", clipped_norm)  # 本例应约为 0.1。
    print("小模型实际参数更新长度：", update_norm)  # 本例应约为 0.01。
    print("裁剪后更新前后损失：", clip_loss_before, clip_loss_after)  # 观察一次裁剪更新的效果。
    print("全局范数、梯度裁剪与单步更新检查：通过")  # 手算、边界情况和完整模型演示均通过后显示。
```

### 参数、效率与老师预检

没有新增模型参数。设总参数数 P，全局范数与统一缩放均需 O(P) 时间，返回裁剪梯度副本另需 O(P) 空间。真实模型 P=719723，float32 裁剪结果另占 2878892 字节，不含计算平方和时的临时双精度数组和其他中间值。本阶段以清晰、可检查为主，后续可以讨论就地缩放与减少临时数组。

老师将以上代码与现有脚本在内存中组合，全部原有检查及新增断言通过；另核对 float32/float64 的形状、类型、返回数组与原输入不共享内存、阈值小于/等于/大于范数、零梯度以及异常非有限梯度被拒绝。以下为老师预检结果，学习者运行结果待补。

```text
演示裁剪前范数： 5.0
演示缩放比例： 0.5
演示裁剪后范数： 2.5
小模型原始梯度范数： 0.4370134224566612
小模型裁剪缩放比例： 0.22882592355596823
小模型裁剪后梯度范数： 0.1
小模型实际参数更新长度： 0.010000000000000002
裁剪后更新前后损失： 1.3442277271788379 1.33987694724801
全局范数、梯度裁剪与单步更新检查：通过
```

运行 python scripts/learn/07_tiny_rnn.py，贴出新增输出并回答：一、梯度为 [3,4]，全局阈值 C=1，缩放比例与裁剪后梯度分别是多少？二、原梯度长度为 0.5，阈值 C=1，是否要把梯度放大到长度 1？完成后检查实际脚本、记录回答与结果、更新 CI 说明并同步 GitHub。

### 第二十二课实操复查：通过，范数裁剪概念继续讲解

老师检查学习者实际脚本及 diff，重新运行，原有检查与本课全部断言通过，未改学习者代码。演示原范数 5、缩放比例 0.5、裁剪后范数 2.5；小模型原始梯度范数 0.4370134224566612，缩放比例 0.22882592355596823，裁剪后范数 0.1，实际参数更新长度 0.010000000000000002；更新前后损失为 1.3442277271788379 和 1.33987694724801。CI 已运行该脚本，本次补充其说明。

口述回答“我没懂”，因此只记录代码实操通过，范数、阈值与比例继续拆开解释，不推进下一课，不追加代码。

先区分三个量：原长度是这批全部梯度合起来有多长；阈值是允许的最大长度；缩放比例是每个梯度分量共同乘的系数。阈值是上限，不是必须达到的目标。范数针对梯度，不是参数本身的大小，也不是损失值。

第一题具体计算：梯度 [3,4]，长度 sqrt(3²+4²)=5；上限 1，小于原长度，所以把长度从 5 缩到 1，所有分量都乘 1/5=0.2。结果 [3*0.2,4*0.2]=[0.6,0.8]；检验 sqrt(0.6²+0.8²)=1。不是把两个分量分别设为 1，也不是每个分量都减去同一个数。

第二题：原长度 0.5，上限 1，没有超过上限，所以比例取 1，原样保留，长度仍是 0.5。不使用 1/0.5=2 将它放大，因为缩放公式仅在超过上限时启用。

下一次只确认一个判断：原梯度长度为 0.5，允许的最大长度为 1，这个梯度超过上限了吗？等待学习者回答，不提前记为概念掌握。

### 阈值比较复核：回答正确

学习者回答“没超过”，正确：0.5<1，所以缩放比例取 1，保留原梯度，长度仍为 0.5；阈值是允许的上限，不是必须达到的目标。本次只确认低于阈值的判断，不将此回答扩展为已独立掌握超阈值时的比例计算。

下一次用超阈值例子继续巩固：原梯度长度为 5，要缩到上限 1，各个梯度分量应该乘 1/5=0.2，还是乘 5？本次不新增代码，第二十二课代码实操已通过，缩放计算继续复核。

### 超阈值缩放比例复核：回答正确

学习者回答“应该乘 0.2”，正确：原长度为 5、上限为 1 时，比例为 1/5=0.2，全部梯度分量使用相同系数，整体长度成为 5*0.2=1。对于梯度 [3,4]，结果为 [0.6,0.8]。没有超过阈值时比例仍为 1，不放大小梯度。

第二十二课代码实操与断言检查通过；低于阈值时保持不变、超过阈值时共同缩小的口述复核均回答正确。后续在训练循环中继续巩固完整求梯度→全局裁剪→统一更新的顺序。本次仅补记回答，未修改练习代码，尚未启动正式语料训练。

## 第二十三课：接入真实语料批次，完成一次训练更新

状态：老师已在内存中预检以下完整脚本，待学习者新建 scripts/learn/10_train_one_batch.py 并亲自输入、运行与回答。没有代写学习者脚本，没有保存模型权重，尚未进行多批训练。本课将第六课真实训练批次与实际 TinyRNN 连接，完成一次训练更新。

### 本课只连接已经学过的部件

真实训练章节→固定词表编码→章节内窗口→一个 NumPy 批次→平均交叉熵和完整梯度→一次全局裁剪→统一更新→同批次重算损失。B=16、T=32，共 512 个预测位置，(p-q)/(B*T) 已在 loss_and_grad 中处理，本脚本不再除一次。模型从随机种子 42 的初始参数开始，仍为 V=5547、H=64、719723 个参数。

固定词表、训练文件 SHA-256 匹配和章节边界继续复用第六课逻辑，不重建词表，不拼接章节。本课取首批顺序窗口；对于当前数据，这 16 个窗口都来自首条训练章节 三国演义:010，起点为 0、32、64、……、480，它不是整套训练数据的均匀随机样本，只用于检查真实数据能否走通一次更新。

### 第一块：新文件中导入模块、取一个真实批次

```python
from importlib import import_module  # 按模块文件名导入已有代码，支持数字开头的文件名。
import numpy as np  # 导入 NumPy，检查编号数组、参数和损失。


def main():  # 定义一次真实训练批次的演示入口。
    data = import_module("06_make_windows")  # 复用已有训练记录、固定词表、窗口和批次函数，并执行该模块顶层的旧检查。
    network = import_module("07_tiny_rnn")  # 复用 TinyRNN、交叉熵与梯度裁剪，该模块的 main 不会被自动执行。
    batch_size = 16  # 最多收集 16 个窗口作为本次训练批次。
    seq_length = 32  # 每个窗口预测 32 个位置。
    learning_rate = 0.1  # 本次演示使用的普通梯度下降学习率。
    max_norm = 1.0  # 将所有参数梯度的总长度限制在 1 以内。
    windows = data.iter_windows(data.train_records, seq_length, seq_length)  # 从真实训练章节按顺序产生窗口，窗口不跨章节。
    batch = next(data.iter_batches(windows, batch_size))  # 只取第一个批次，不遍历训练更新整个数据集。
    x, y = batch["x"], batch["y"]  # 取出真实输入编号和向后错开一位的目标编号。
    assert x.shape == y.shape == (batch_size, seq_length), "本例首批尺寸不符合预期"  # 当前数据足够组成 16 行的完整首批。
    assert x.dtype == y.dtype == np.int64 and np.array_equal(x[:, 1:], y[:, :-1]), "真实批次错位或类型不正确"  # 再次核对接入模型的数据。
    model = network.TinyRNN(vocab_size=len(data.tokens), hidden_size=64, seed=42)  # 用真实固定词表创建全新的随机模型。
```

文件名以数字开头，普通 from 07_tiny_rnn import ... 不是合法 Python 语句，使用标准库 import_module("07_tiny_rnn") 按字符串导入。按 python scripts/learn/10_train_one_batch.py 启动时，Python 会把脚本所在目录加入模块搜索路径，所以可以找到同目录的 06 和 07，不要求手动改 sys.path。

06_make_windows.py 目前包含顶层读取、输出与完整窗口统计，首次导入会先执行已有数据检查，所以终端前半段会显示旧的第六课输出；这些检查不会训练模型。07_tiny_rnn.py 已有 main 保护，因此导入它不会运行旧模型演示。本课 main 内才导入数据模块；后续整理数据模块时再将演示移入 main，本课不同时展开大段结构调整。

### 第二块：接在 main 内，求梯度、裁剪并更新一次

```python
    params_before = {name: param.copy() for name, param in model.params.items()}  # 保存本次前向使用的参数，用来检查反向不会更新它们。
    loss_before, raw_grads = model.loss_and_grad(x, y)  # 对 512 个真实预测位置完成前向、平均损失和全部反向。
    assert all(np.array_equal(param, params_before[name]) for name, param in model.params.items()), "求梯度过程中修改了参数"  # 完整反向结束前参数应保持不变。
    clipped_grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 汇总完成后，对五组梯度做一次全局裁剪。
    clipped_norm = network.gradient_norm(clipped_grads)  # 得到本次实际用于更新的梯度总长度。
    assert clipped_norm <= max_norm + 1e-6, "裁剪后的梯度超过阈值"  # 为 float32 数组的舍入误差保留容差。
    for name in model.params:  # 使用同一批计算出的完整裁剪梯度更新五组参数。
        model.params[name] -= learning_rate * clipped_grads[name]  # 每组参数都减去学习率乘对应梯度。
    assert all(np.isfinite(param).all() for param in model.params.values()), "更新后参数出现异常"  # 确认更新结果仍是有限数值。
    assert any(not np.array_equal(param, params_before[name]) for name, param in model.params.items()), "本次更新没有改变任何参数"  # 确认真实执行了一次参数更新。
    loss_after, _ = network.cross_entropy(model.forward(x), y)  # 使用更新后的参数，对同一个批次重新前向计算损失。
    assert np.isfinite(loss_after) and loss_after < loss_before, "本例单批更新没有降低损失"  # 验证当前数据、种子和学习率的具体结果。
```

这里更新前后的损失使用同一批 x、y，但必须重新前向计算，不能复用旧 logits。params_before 是参数副本，用来检查求梯度不改参数，以及真正执行了更新；它不是保存到磁盘的权重文件。assert loss_after<loss_before 仅检查这个固定批次、种子和学习率下的具体结果；一般训练不要求每个不同批次的损失都比上一批小。

### 第三块：接在 main 内打印，再顶格添加入口

```python
    print("真实批次来源首行：", batch["chapter_ids"][0], "起点：", batch["starts"][0])  # 显示第一行窗口的章节和零起点。
    print("真实批次 X/Y 形状：", x.shape, y.shape)  # 应为两个 (16,32)。
    print("本批预测位置数：", y.size)  # 应为 16*32=512。
    print("首行输入前八字：", repr("".join(data.tokens[index] for index in x[0, :8])))  # 将前八个输入编号还原成文字。
    print("首行目标前八字：", repr("".join(data.tokens[index] for index in y[0, :8])))  # 展示向后错开一字的目标。
    print("真实批次原始梯度范数：", raw_norm)  # 显示本批完整梯度的总长度。
    print("真实批次裁剪比例：", scale)  # 没超过阈值时比例为 1，超过时小于 1。
    print("真实批次裁剪后范数：", clipped_norm)  # 应不超过阈值 1，允许浮点舍入误差。
    print("真实批次更新前后损失：", loss_before, loss_after)  # 两个数都在同一个训练批次上计算。
    print("真实语料单批训练更新检查：通过")  # 所有检查通过后显示。


if __name__ == "__main__":  # 只有直接运行新脚本时才开始本次演示。
    main()  # 运行一次真实批次更新。
```

### 学习结论与效率

本批输入和目标均为 (16,32)，类型 int64，两数组数据区共 8192 字节；状态形状 (16,32,64)，float32 数据区为 131072 字节；单份 logits 或概率形状 (16,32,5547)，含 2840064 个 float32，数据区为 11360256 字节。这只是单个数组的大小，不是总内存峰值；损失和反向还有其他临时数组、参数、梯度、副本。

一次前向和反向的主要计算量约 O(B*T*(H²+H*V))。本课额外重算一次前向用于比较更新后损失。窗口内使用真实前一字输入进行下一字预测，完成一次带真实目标的训练更新；同批损失下降只能说明这个批次上的目标改善，不能据此判断验证集表现或生成质量。验证集与测试集仍不参与本次更新。

本脚本暂不保存权重，每次运行都从种子 42 的全新随机模型重新开始，所以重复运行两次不等于接着训练两步。当前数据和 NumPy 环境下结果可复现，跨平台浮点尾数允许差异。本课学习率 0.1、裁剪阈值 1 是教学演示配置，不宣称已通过验证集选择。

### 老师预检输出与待提交

老师在内存中执行以上脚本两次，所有断言通过，固定种子与数据产生相同结果；未创建学习者脚本、未保存权重，未改训练数据。首次导入会先输出旧数据检查，新增结果出现在末尾：

```text
真实批次来源首行： 三国演义:010 起点： 0
真实批次 X/Y 形状： (16, 32) (16, 32)
本批预测位置数： 512
首行输入前八字： '第十回\u3000勤王室马'
首行目标前八字： '十回\u3000勤王室马腾'
真实批次原始梯度范数： 0.320137853404229
真实批次裁剪比例： 1.0
真实批次裁剪后范数： 0.320137853404229
真实批次更新前后损失： 8.648512840270996 8.638714790344238
真实语料单批训练更新检查：通过
```

运行 python scripts/learn/10_train_one_batch.py，提交末尾新增输出。回答：一、本批 16 个窗口，每个 32 个预测位置，平均交叉熵一共对多少个位置求平均？二、同一训练批次损失下降，能否说明验证集效果也一定变好？完成后检查实际脚本、补记结果、将新脚本加入 CI 并完整同步 GitHub。

### 第二十三课实操与问答复查：通过

老师检查学习者实际新建的 scripts/learn/10_train_one_batch.py 并重新运行；已有数据检查及真实单批训练更新断言全部通过，未修改学习者代码。首行来源 三国演义:010，起点 0，X/Y 均为 (16,32)，本批 512 个预测位置，输入与目标错位正确。

实际原始梯度范数为 0.320137853404229，小于阈值 1，所以裁剪比例为 1，裁剪后范数保持原值；同批平均损失从 8.648512840270996 降到 8.638714790344238。该结果与预检一致。新脚本加入 GitHub Actions 已完成课程检查中，后续推送会自动运行其断言。

第一题学习者回答“512个”，正确：平均交叉熵覆盖 16*32=512 个预测位置。第二题回答“不一定”，正确：训练批次与验证集是不同数据，同批损失下降不保证验证集损失下降，不能据此宣称泛化表现或生成质量提高。

当前进度：已用真实语料完成一次前向、完整反向、全局裁剪判断与参数更新。权重只存在于本次进程内，尚未保存；每次重新运行脚本都会从固定随机种子的全新模型开始。下一课可学习让同一个模型连续处理多个批次，并区分 step 与 epoch，再逐步加入验证、日志与权重保存；这些后续功能尚未实现。

## 第二十四课：同一个模型连续训练多个批次，区分 step 与 epoch

状态：老师已在内存中预检以下代码，待学习者新建 scripts/learn/11_train_steps.py 并手敲、运行与回答。未创建学习者脚本或保存模型。本课只连续训练前五个批次，尚未完成一整轮、多轮训练、验证或权重持久化。

### step、batch 与 epoch

batch 是一次送入模型的一组窗口；本课一个训练 step 是对一个 batch 完成前向、反向、裁剪和一次统一参数更新。五组参数各更新一遍合起来算一步，不算五步。epoch 是完整遍历当前训练窗口集一遍。

当前窗口数 76584，批大小 16，保留尾批，所以 steps_per_epoch=ceil(76584/16)=4787，其中 4786 个完整批次和一个八行尾批。五步只处理 80 个窗口、2560 个预测位置，不能称为五轮，也没有完成一整轮。这里按当前无梯度累积的普通训练循环定义更新步数；以后若引入梯度累积需再区分微批次与优化器步数。

连续训练参数递推：g_k=∇L_(batch_k)(θ_(k-1))，裁剪后 θ_k=θ_(k-1)-η*clip(g_k)。第二批使用第一批更新后的参数，以此类推。模型只在循环外创建一次，批次生成器也只创建一次。不要在循环里重新初始化模型，否则丢弃前面的学习；不要每步重建批次生成器再 next，否则反复取首批。

模型参数跨批保留，与隐藏状态跨批保留是两件事。当前 forward_hidden 每条窗口都从零状态开始，不把上一批隐藏状态带给下一批；窗口可能来自不同位置或章节，本课保持既定训练定义。

### 第一块：新建文件，准备模型与连续批次流

```python
from importlib import import_module  # 按字符串导入已有数据与模型模块。
from itertools import islice  # 从批次生成器中只取指定数量的批次。
import numpy as np  # 检查批次数组与训练损失。


def main():  # 定义连续训练少量批次的演示入口。
    data = import_module("06_make_windows")  # 复用真实训练章节和批次函数，首次导入会执行旧数据检查。
    network = import_module("07_tiny_rnn")  # 复用模型、完整反向和梯度裁剪。
    batch_size = 16  # 每个批次最多包含 16 个窗口。
    seq_length = 32  # 每个窗口预测 32 个位置，窗口步长也取 32。
    max_steps = 5  # 本课只更新五个批次，不训练完整一轮。
    learning_rate = 0.1  # 每一步使用相同的普通梯度下降学习率。
    max_norm = 1.0  # 每一步都对完整梯度使用相同的全局范数上限。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 只在循环外创建一次模型，之后持续更新同一份参数。
    windows = data.iter_windows(data.train_records, seq_length, seq_length)  # 只创建一次窗口生成器，使窗口位置持续向后移动。
    batches = data.iter_batches(windows, batch_size)  # 把这一个窗口流打包为连续批次。
    completed_steps = 0  # 累计真正完成的参数更新次数。
    seen_windows = 0  # 累计真正用于更新的窗口数。
    seen_positions = 0  # 累计真正参与损失计算的预测位置数。
    loss_sum = 0.0  # 累计各批平均损失乘其位置数，准备计算加权平均。
```

islice(batches,max_steps) 从现有批次流最多取五批，不会把全部批次转换成列表。首次导入第六课模块仍会先输出旧检查，那些数据统计不算模型训练 step。

### 第二块：接在 main 内，连续更新五步

```python
    for batch in islice(batches, max_steps):  # 顺着同一个批次生成器取前五批，每批更新一次。
        x, y = batch["x"], batch["y"]  # 取本批输入和正确的下一字编号。
        assert x.shape == y.shape and 0 < x.shape[0] <= batch_size and x.shape[1] == seq_length, "批次尺寸错误"  # 按实际行数检查，也允许未来遇到不足一批的尾批。
        assert x.dtype == y.dtype == np.int64 and np.array_equal(x[:, 1:], y[:, :-1]), "批次类型或错位错误"  # 保证训练格式仍然正确。
        loss, raw_grads = model.loss_and_grad(x, y)  # 用当前参数计算本批更新前的平均损失与完整梯度。
        assert np.isfinite(loss), "训练损失不是有限数"  # 发现异常数值就停止，不继续污染参数。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 每步完整反向结束后再做一次全局裁剪。
        for name in model.params:  # 使用同一批次算出的五组梯度统一更新参数。
            model.params[name] -= learning_rate * grads[name]  # 本步新参数会被下一批直接使用。
        assert all(np.isfinite(param).all() for param in model.params.values()), "更新后参数异常"  # 确认连续更新没有产生无效参数。
        completed_steps += 1  # 五组参数更新完成，记作一个 step。
        seen_windows += x.shape[0]  # 统计实际行数，不假定永远是完整批次。
        seen_positions += y.size  # 统计本批实际预测位置数。
        loss_sum += loss * y.size  # 将平均损失还原为本批损失总和，再跨批累计。
        print(f"step={completed_steps} loss={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f} 来源={batch['chapter_ids'][0]} 起点={batch['starts'][0]}")  # 显示每步更新前的损失和首行窗口来源。
```

每行日志 loss 都是这一批更新前、当前参数下的平均交叉熵。不同步的数据不同，不能要求日志逐行严格下降；本课没有这样的断言。初始几步都来自首条训练章节，是顺序取数据的演示，不宣称这五批代表所有书籍的平均表现；后面再设计训练顺序打乱和验证。

raw_norm 是裁剪前长度，scale 是当前裁剪系数。参数在每步全部反向完成后才更新，下一步再次完整计算梯度。loss_sum += loss*y.size 按实际预测位置数加权，兼容未来可能遇到的小尾批。权重应按预测位置数而非固定批数求平均。

### 第三块：接在 main 内总结，再添加入口

```python
    assert completed_steps == max_steps and seen_positions > 0, "当前训练数据不足以完成本课五步"  # 本例训练集应能提供足够批次。
    windows_per_epoch = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in data.train_records)  # 根据当前窗口长度与步长，独立计算完整一轮的窗口数。
    steps_per_epoch = (windows_per_epoch + batch_size - 1) // batch_size  # 尾批也更新一次，用向上取整计算一轮步数。
    print("已完成更新步数：", completed_steps)  # 应为 5，而不是五组参数乘五步。
    print("已训练窗口数：", seen_windows)  # 当前五个完整批次共 80 个窗口。
    print("已训练预测位置数：", seen_positions)  # 当前共 2560 个预测位置。
    print("训练过程按位置加权的平均损失：", loss_sum / seen_positions)  # 汇总各步更新前的损失，模型参数在这些步骤间持续变化。
    print("完整一轮需要的更新步数：", steps_per_epoch)  # 当前配置为 4787 步，包含最后八行的尾批。
    print("本次是否遍历完整训练集：", seen_windows == windows_per_epoch)  # 本课只取五批，因此应为 False。
    print("同一模型连续多批更新检查：通过")  # 所有本课断言通过后显示。


if __name__ == "__main__":  # 只有直接运行文件时才开始训练演示。
    main()  # 创建一次模型并连续执行五步更新。
```

训练过程平均损失等于 Σ_k(n_k*L_k)/Σ_k n_k，其中 n_k 为实际位置数。注意 θ 在步骤间持续变化，该均值只是这段训练过程的记录，不是最后模型固定参数下的训练集评估，更不是验证损失。最后模型的评估需要另做一次固定参数的前向统计。

### 复现、效率与老师预检

仍不保存权重，重新启动脚本会重新初始化固定种子模型；同一次运行内则确实连续保留并更新参数。当前数据、顺序和环境相同时可重复得到同样结果，跨平台允许浮点尾数差异。

模型参数仍为 719723，每步的数据大小仍为 B=16、T=32。五步顺序运行不会同时保存五批全部中间状态；每步主计算约 O(B*T*(H²+HV))，五步约为五倍单步主要训练计算，不包含导入时数据统计等固定开销。本课每步不额外重算同批更新后损失，因此与上一课含额外前向的用时不能直接当作同等操作比较；未进行实际性能基准测试。

老师在内存中运行上述源码两次，全部断言通过，五步顺序来源与数值一致。未替学习者写入脚本、未保存模型权重。新结果在旧检查之后：

```text
step=1 loss=8.648513 grad_norm=0.320138 scale=1.000000 来源=三国演义:010 起点=0
step=2 loss=8.656752 grad_norm=0.666002 scale=1.000000 来源=三国演义:010 起点=512
step=3 loss=8.613806 grad_norm=0.392007 scale=1.000000 来源=三国演义:010 起点=1024
step=4 loss=8.647432 grad_norm=0.416489 scale=1.000000 来源=三国演义:010 起点=1536
step=5 loss=8.615304 grad_norm=0.471498 scale=1.000000 来源=三国演义:010 起点=2048
已完成更新步数： 5
已训练窗口数： 80
已训练预测位置数： 2560
训练过程按位置加权的平均损失： 8.63636131286621
完整一轮需要的更新步数： 4787
本次是否遍历完整训练集： False
同一模型连续多批更新检查：通过
```

运行 python scripts/learn/11_train_steps.py，提交五行 step 日志与末尾统计。回答：一、同一模型连续完成五个批次更新，算 5 个 step 还是 5 个 epoch？二、为什么 model = TinyRNN(...) 必须放在本课训练循环外面？完成后复查实际脚本、补充笔记、加入 CI 并完整同步 GitHub。

### 第二十四课实操与问答复查：通过

老师检查学习者实际新建脚本并重新运行，全部旧数据检查与本课断言通过，未修改学习者代码。五步日志损失依次为 8.648513、8.656752、8.613806、8.647432、8.615304；各步来源首行起点依次为 0、512、1024、1536、2048，说明批次流持续向后移动。所有日志中的损失都是各批在对应更新前参数下的损失，没有要求不同批次之间单调下降。

实际完成 5 步更新、80 个窗口、2560 个预测位置，训练过程按位置加权的平均损失为 8.63636131286621；完整一轮需要 4787 步，本次没有遍历完整训练集。新脚本已加入 GitHub Actions 已完成课程检查。

第一题学习者回答“五个批次是5个step”，正确。本课一个 batch 对应一次参数更新，所以是 5 个 step，不是 5 个 epoch。第二题回答模型只需加载一次，循环内重载会无法有效学习，核心理解正确。术语补充：本代码 TinyRNN(...) 是创建模型并随机初始化参数，尚未从权重文件加载；放在循环内会每批重新初始化，丢掉前面训练结果。固定 seed=42 时每次甚至会回到同一份初始参数。训练循环的职责是取新批次、求梯度和更新同一个模型，更新步数统计只是其中一部分。

第二十四课实操与问答通过。同一次运行内参数连续保留，隐藏状态仍按每个窗口从零开始；重新运行整个脚本仍会从新模型开始，尚未保存权重。下一课可增加固定验证数据上的前向评估，区分训练过程损失和固定模型的验证损失；该评估尚未实现。

## 第二十五课：固定验证子集，只做前向评估

状态：老师已在内存中预检以下修改，待学习者亲自修改 scripts/learn/11_train_steps.py、运行并回答。未替学习者修改练习脚本；本课在已有五步训练前后评估同一份验证子集，尚未评估完整验证集、使用测试集或保存模型。

### 训练与评估的区别

训练：前向→损失→反向→裁剪→更新。验证：前向→损失→按位置汇总，参数保持不变。验证文字使用训练集建立的固定词表，未见字符继续映射到 <UNK>，不把验证字符加入词表。这里模型是纯 NumPy，没有自动求导，也没有 dropout 或 batch normalization；不需要为这个模型调用不存在的 eval() 或 no_grad() 接口。

固定参数 θ 时，验证平均交叉熵 L_val(θ)=Σ_b(n_b*L_b(θ))/Σ_b n_b，其中 n_b 是每个批次实际预测位置数。评估期间所有批次使用同一份 θ；这与训练日志在不同参数点的平均值含义不同。

### 第一块：文件顶部增加一个导入

```python
import json  # 导入 JSON 工具，读取独立的验证章节文件。
```

原来的 import_module、islice 和 np 导入保留。

### 第二块：在 main 前顶格增加评估函数

```python
def evaluate(model, batches, network):  # 固定参数，只前向计算给定批次的平均损失。
    saved_params = {name: param.copy() for name, param in model.params.items()}  # 保存评估前的参数，用来检查评估没有更新模型。
    loss_sum = 0.0  # 累计所有预测位置的损失总和。
    position_count = 0  # 累计实际参与评估的预测位置数。
    for batch in batches:  # 依次评估固定子集中的各个批次。
        x, y = batch["x"], batch["y"]  # 取验证输入与正确的下一字编号。
        loss, _ = network.cross_entropy(model.forward(x), y)  # 只做前向和损失计算，不调用反向传播。
        assert np.isfinite(loss), "验证损失不是有限数"  # 出现无效结果时停止检查。
        loss_sum += loss * y.size  # 按本批实际位置数加权，不能把大小不同的批次简单等权平均。
        position_count += y.size  # 累计包括小尾批在内的实际位置数。
    assert position_count > 0, "没有可评估的位置"  # 防止空子集导致除以零。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "评估修改了参数"  # 参数必须与评估之前完全一致。
    return loss_sum / position_count, position_count  # 返回固定参数下的加权平均损失和评估规模。
```

函数只调用 model.forward 和交叉熵，不计算参数梯度，也不裁剪或更新参数。保存参数副本并逐元素比较，是本课用来核对“评估不修改模型”的教学检查，不能把它理解为评估的训练步骤。若原始参数已含 NaN，前向或损失检查会先报错。

### 第三块：在 main 内模型创建后、训练窗口生成前插入

找到 model=network.TinyRNN(...)，紧接它添加以下代码，旧 windows=... 及之后的训练循环仍保留在新代码后。此时 batch_size、seq_length 和模型都已定义，尚未执行任何训练更新。

```python
    validation_path = data.ROOT / "data" / "splits" / "validation.jsonl"  # 读取此前按完整章节划分的验证集。
    with validation_path.open(encoding="utf-8") as file:  # 以 UTF-8 打开验证章节文件。
        validation_records = [json.loads(line) for line in file]  # 逐行还原验证章节，保留原有顺序。
    assert not ({record["id"] for record in data.train_records} & {record["id"] for record in validation_records}), "训练与验证章节标识重叠"  # 确认评估章节没有进入训练集。
    validation_windows = data.iter_windows(validation_records, seq_length, seq_length)  # 继续使用训练词表编码验证文字，未见字符映射到已有未知编号。
    validation_batches = list(islice(data.iter_batches(validation_windows, batch_size), 3))  # 固定保存前三个验证批次，前后评估复用相同数据。
    assert len(validation_batches) == 3, "本课验证数据不足三批"  # 当前数据应能提供三批窗口。
    validation_before, validation_positions = evaluate(model, validation_batches, network)  # 在任何训练更新之前评估初始模型。
    print("固定验证子集批次数：", len(validation_batches))  # 应为 3，不能称为整个验证集。
    print("固定验证子集预测位置数：", validation_positions)  # 当前三批共有 1536 个位置。
    print("训练前固定验证子集损失：", validation_before)  # 记录未训练模型在固定验证子集上的损失。
```

把前三个验证批次保存成列表，而不是将整个验证集窗口全部装入列表。前后都遍历同一个小列表，不会因生成器耗尽导致第二次评估为空；也不会在两次评估中换不同的窗口。当前 B=16、T=32，三批共 48 个窗口、1536 个位置，输入和目标数组合计约 24 KiB，不含元数据与其他数组。

这只是验证集顺序前缀，并非随机代表样本或整个验证集的准确估计；对于当前排序，前缀来自验证集中较早的章节，不覆盖所有书籍。所有输出明确写“固定验证子集”。后续扩大评估时，再说明全验证集覆盖和计算成本。

### 第四块：在原 main 末尾追加训练后评估

放在原 print("同一模型连续多批更新检查：通过") 后，仍在 main 内，缩进四格；文件末尾 if __name__ 入口仍顶格保留。

```python
    validation_after, after_positions = evaluate(model, validation_batches, network)  # 五步训练结束后，用同一子集评估最终参数。
    assert after_positions == validation_positions, "前后评估位置数不一致"  # 前后比较必须使用相同的数据范围。
    print("训练后固定验证子集损失：", validation_after)  # 不要求它一定低于训练前，验证结果需要如实记录。
    print("固定验证子集损失变化：", validation_after - validation_before)  # 负数表示该子集损失降低，正数表示升高。
    print("验证只前向、参数不变与固定子集检查：通过")  # 正确性检查通过不代表验证效果一定改善。
```

不要增加 assert validation_after<validation_before。数据流程正确和验证效果改善是不同判断：如果验证损失升高，仍然要如实输出并分析。也不要将训练过程平均损失和验证损失的差值直接解释为严谨的最终模型泛化差距，因为训练过程平均值来自不断变化的参数。

### 理解结果与评估范围

训练前后使用同一批验证输入和标签，所以损失变化主要用于比较两份参数在这个固定子集上的表现。下降只说明该子集平均损失改善，不保证完整验证集、其他领域或文字生成质量都改善。验证数据不直接用于参数梯度，但以后选择学习率、训练步数或模型时仍会间接影响选择，因此测试集继续保留，不在本课使用。

训练时仍只迭代 data.train_records，验证记录只传给评估批次构造。章节 id 交集为空是本课的接口核对，不代表已经证明没有任何重复段落。模型每个窗口的初始隐藏状态仍为零，训练和验证沿用同一计算定义。

### 效率与老师预检

训练仍为五步，验证在训练前后各做三批前向，共六批额外前向，不做反向。主要验证计算量约 O(B*T*(H²+HV)) 每批；不会为验证生成五组梯度。教学用参数快照额外占约 2.88 MB float32 数据区，后续可在可信实现中减少该检查开销。当前子集只缓存三批编号，仍逐批计算分数和状态。

老师在内存中组合修改并运行全部旧检查与新增断言，实际模型两次评估参数均不改变；另用两批大小不同的固定损失检查按位置加权（两位置损失均为 2、另一位置损失为 4，正确均值为 8/3，而不是 3），并检查空评估会报错。以下为老师预检，学习者运行结果待补：

```text
固定验证子集批次数： 3
固定验证子集预测位置数： 1536
训练前固定验证子集损失： 8.651848157246908
step=1 loss=8.648513 grad_norm=0.320138 scale=1.000000 来源=三国演义:010 起点=0
step=2 loss=8.656752 grad_norm=0.666002 scale=1.000000 来源=三国演义:010 起点=512
step=3 loss=8.613806 grad_norm=0.392007 scale=1.000000 来源=三国演义:010 起点=1024
step=4 loss=8.647432 grad_norm=0.416489 scale=1.000000 来源=三国演义:010 起点=1536
step=5 loss=8.615304 grad_norm=0.471498 scale=1.000000 来源=三国演义:010 起点=2048
已完成更新步数： 5
已训练窗口数： 80
已训练预测位置数： 2560
训练过程按位置加权的平均损失： 8.63636131286621
完整一轮需要的更新步数： 4787
本次是否遍历完整训练集： False
同一模型连续多批更新检查：通过
训练后固定验证子集损失： 8.628433227539062
固定验证子集损失变化： -0.023414929707845644
验证只前向、参数不变与固定子集检查：通过
```

运行 python scripts/learn/11_train_steps.py，提交固定验证子集批次数、位置数、训练前后损失和变化值；原五步日志也应继续一致。回答：一、验证时应调用 model.forward，还是调用 loss_and_grad 后再更新参数？二、前后比较验证损失，为什么本课要使用同一份固定子集？完成后复查脚本、记录回答与结果、更新 CI 说明并同步 GitHub。

### 第二十五课实操复查：通过，验证接口口述继续纠正

老师检查学习者实际脚本与 diff，并重新运行；已有数据检查、连续五步训练与新增验证检查全部通过。实际代码 evaluate 正确调用 model.forward 与交叉熵，没有反向或更新。固定验证子集为 3 批、1536 个位置，训练前损失 8.651848157246908，训练后 8.628433227539062，变化 -0.023414929707845644。评估前后参数完全一致，原五步训练日志也与上一课一致。

老师仅清理行尾空白并补齐文件末尾换行，清理前后语法树一致，未修改计算逻辑。已有 CI 运行该脚本，本次补充固定验证子集检查的说明。

第二题学习者回答“控制变量”，正确：前后固定验证输入、标签和评估定义，比较同一数据上两份参数的表现；如果换数据，损失变化会混入数据难度差异。下降只证明当前子集平均损失改善，不推广为完整验证集或生成质量必然改善。

第一题回答“调用 loss_and_grad()”，需要澄清本课接口选择。forward 只计算预测分数，再由 cross_entropy 得到评估损失；loss_and_grad 则额外执行完整反向，返回五组梯度，评估不需要这些梯度。必须准确区分：本项目单独调用 loss_and_grad 并不会自动更新参数；如果忽略其梯度，只读取损失，数值上仍可用于评估，但会多做不必要的反向与梯度分配。不能说只要调用该方法就已经更新或发生了数据泄漏；真正错误的是随后用验证梯度更新模型，那会使验证数据参与参数训练。

因此本课推荐且实际采用的流程为 logits=model.forward(x)；loss,_=cross_entropy(logits,y)，随后仅统计损失。代码实操与第二题通过，第一题接口区别待确认，不推进下一课。下一次只确认：验证批次计算完损失后，是否应该执行“参数减学习率乘梯度”的更新语句？

### 验证不更新参数复核：回答正确

学习者回答“因为是在做验证，所以不需要更新参数”，正确。评估保持当前参数固定，只前向计算预测分数与损失并汇总；本课使用 forward 加 cross_entropy，不额外计算反向梯度。训练批次才用于完整反向、裁剪与参数更新。

第二十五课实操检查、固定子集控制变量问答和验证不更新参数复核均通过。保留之前的接口澄清：loss_and_grad 单独调用不会自动更新，但会多算评估不需要的梯度。当前仅验证固定前三批，未扩大为完整验证集；本次只补记回答，未修改练习代码。下一课可学习保存五组参数和模型配置，并重新加载核对预测与损失，使训练成果在退出程序后仍可使用；该功能尚未实现。

## 第二十六课：保存模型权重与词表，重新加载核对结果

状态：老师已在临时目录预检以下修改，待学习者亲自修改 scripts/learn/11_train_steps.py 并运行。未代写学习者代码，未在仓库中生成本课权重。本课实际运行后才会保存 model/learn_rnn_step005.npz，不覆盖原 model/tiny_rnn.npz 示例。

### 需要保存什么

训练结果主要是五组浮点参数，重新创建随机模型并不能恢复训练成果。还需保留隐藏维度、模型结构格式、词表编号顺序与已完成训练步数。本课将整个已有 vocab 字典一起保存，包括 tokens、未知字符配置和训练数据来源 SHA-256；词表顺序与权重必须配套，否则同一编号会被解释成其他字符。

保存采用 np.savez_compressed：文件包含五组有名字的 NumPy 数组，加一个 JSON 字符串数组 metadata。加载时 allow_pickle=False，不保存或加载整个 Python 模型对象。文件仍需搭配本项目 TinyRNN 实现，它不是独立可执行程序，也不是其他示例脚本权重格式的自动兼容文件。

### 第一块：顶部增加路径导入

```python
from pathlib import Path  # 导入路径工具，创建权重目录并检查文件大小。
```

保留已有 json、np 和其他导入。

### 第二块：在 main 前顶格添加保存与加载函数

```python
def save_checkpoint(path, model, vocab, completed_steps):  # 将模型参数、词表、结构配置和已完成步数保存到一个文件。
    path = Path(path)  # 把传入路径统一转换为 Path 对象。
    assert path.suffix == ".npz", "本课权重文件使用 .npz 后缀"  # 避免保存工具自动补后缀导致路径不一致。
    assert len(vocab["tokens"]) == model.vocab_size, "词表与模型大小不一致"  # 每个输入输出编号都必须对应保存的词表。
    assert isinstance(completed_steps, int) and completed_steps >= 0, "训练步数不合法"  # 已完成更新次数必须是非负整数。
    assert all(param.dtype == np.float32 and np.isfinite(param).all() for param in model.params.values()), "本课只保存有限的 float32 参数"  # 与当前正式模型的类型保持一致。
    metadata = {"format": "tiny_rnn_char_v1", "hidden_size": model.hidden_size, "completed_steps": completed_steps, "vocab": vocab}  # 保存重建模型和解释编号所需的信息。
    metadata_text = json.dumps(metadata, ensure_ascii=False)  # 将配置转成普通 JSON 字符串，保留中文字符。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保权重文件的父目录存在。
    np.savez_compressed(path, metadata=np.array(metadata_text), **model.params)  # 将字符串配置和五组参数压缩到同一个 npz 文件。


def load_checkpoint(path, network):  # 从权重文件创建一个带已训练参数的新模型。
    with np.load(path, allow_pickle=False) as archive:  # 只读取数组和字符串，不依赖 Python 对象反序列化。
        metadata = json.loads(str(archive["metadata"].item()))  # 将保存的单个字符串数组还原为配置字典。
        assert metadata["format"] == "tiny_rnn_char_v1", "权重格式与本模型不匹配"  # 不把其他结构的权重当成本课模型。
        vocab = metadata["vocab"]  # 使用权重文件内保存的词表，保留原来的编号顺序。
        model = network.TinyRNN(len(vocab["tokens"]), hidden_size=metadata["hidden_size"], seed=0)  # 按保存的维度创建模型，随机初值稍后全部被覆盖。
        for name, expected in model.params.items():  # 逐一加载模型必须具有的五组参数。
            value = archive[name]  # 取出当前参数数组。
            assert value.shape == expected.shape and value.dtype == expected.dtype and np.isfinite(value).all(), name + "权重形状、类型或数值不正确"  # 防止错误数组进入模型。
            model.params[name] = value.copy()  # 复制已训练参数，文件关闭后新模型仍可独立使用。
    return model, vocab, metadata["completed_steps"]  # 返回新模型、对应词表和保存时的训练步数。
```

metadata 中 format 标明本课字符 RNN 结构版本；hidden_size 配置隐藏维；词表长度确定 V；每个参数按名称检查形状、float32 类型和有限数值。加载时先创建同结构模型，再完整替换五组随机参数，所以构造时 seed=0 不决定加载后的训练结果。value.copy() 保证数组由模型独立持有。加载过程不会再次训练。

`model.params` 将键映射到数组，`**model.params` 在函数调用中把字典展开为同名关键字参数，即 embedding=...、recurrent=...、bias=...、output=...、output_bias=...。metadata 是独立的字符串数组；JSON 能保存普通整数、列表和字典，权重数组则由 NumPy 自己保存，不转成庞大的 JSON 数值列表。

本课只支持正式 float32 模型文件。当前普通梯度下降没有额外动量状态，但仅保存权重与步数仍不等于精确恢复整个训练过程：文件未保存批次生成器位置、未来随机打乱状态等；本课不实现断点续训。

### 第三块：在原 main 末尾追加保存与重载检查

放在上一课 print("验证只前向、参数不变与固定子集检查：通过") 后面，保持四格缩进，文件末尾入口判断仍顶格。沿用现有 model、data、completed_steps、validation_batches、validation_after、after_positions。

```python
    checkpoint_path = data.ROOT / "model" / "learn_rnn_step005.npz"  # 将本课五步训练成果保存在独立文件，不覆盖原示例权重。
    save_checkpoint(checkpoint_path, model, data.vocab, completed_steps)  # 保存训练后的五组参数与完整词表配置。
    loaded_model, loaded_vocab, loaded_steps = load_checkpoint(checkpoint_path, network)  # 从磁盘加载成一个新的模型对象。
    assert loaded_model is not model and loaded_vocab == data.vocab and loaded_steps == completed_steps, "加载配置或训练步数不一致"  # 新对象应保留原词表和步数。
    assert all(np.array_equal(param, loaded_model.params[name]) for name, param in model.params.items()), "加载后的参数发生变化"  # 无损保存后五组参数应逐元素完全一致。
    probe_x = validation_batches[0]["x"]  # 固定同一个验证输入，比较加载前后的预测分数。
    logits_before = model.forward(probe_x)  # 用内存中的训练模型计算分数。
    logits_loaded = loaded_model.forward(probe_x)  # 用磁盘重新加载的模型计算相同输入的分数。
    assert np.allclose(logits_before, logits_loaded, rtol=1e-5, atol=1e-6), "加载前后预测不一致"  # 在浮点容差内检查预测计算一致。
    loaded_loss, loaded_positions = evaluate(loaded_model, validation_batches, network)  # 重新评估加载模型，不进行训练。
    assert loaded_positions == after_positions and np.isclose(loaded_loss, validation_after, rtol=1e-6, atol=1e-7), "加载前后验证损失不一致"  # 使用相同固定子集比较损失。
    assert checkpoint_path.stat().st_size < 1_000_000_000, "权重文件超过本项目的 1 GB 上限"  # 检查实际保存文件的字节数。
    print("权重保存路径：", checkpoint_path)  # 显示可供以后独立加载的文件位置。
    print("权重文件字节数：", checkpoint_path.stat().st_size)  # 文件包含压缩参数与配置，大小不等于裸参数字节数。
    print("加载后训练步数：", loaded_steps)  # 应为 5，这只是保存的记录，不会自动执行继续训练。
    print("加载前后验证损失：", validation_after, loaded_loss)  # 两个值应在容差内一致。
    print("模型保存、重载与预测一致性检查：通过")  # 配置、参数、预测与损失检查全部通过后显示。
```

npz 对数组是无损保存，参数必须逐元素完全一致；同输入的浮点预测与损失按容差核对，不能只验证“文件存在”就认为加载正确。重新加载的模型是新对象，参数来源是文件，不靠旧对象的引用。

### 磁盘大小、仓库管理与边界

裸参数数据区为 719723*4=2878892 字节，npz 内还有词表与配置及压缩容器开销，压缩后大小不等于裸参数大小。本课文件远小于用户要求的 1 GB，代码也检查实际文件字节数。加载后的第二个模型需要另一份参数内存，比较预测也会临时分配分数数组，不把文件大小当作运行内存峰值。

保存函数会将相同路径替换为本次五步训练结果，旧示例权重不受影响；目前脚本每次重新运行仍从初始模型训练五步再保存，尚未实现从上次文件接着训练。接下来的文字生成脚本可以直接加载该文件。

当前 .gitignore 默认忽略 model 内的新训练产物。为了遵守每课完成后完整上传的约定，待本课实操完成后，将这个约数 MB 的学习里程碑权重作为明确例外连同代码笔记一起上传，并记录大小；不把它与未来大量实验检查点混为一谈。本次备课不修改忽略规则或上传尚未生成的权重。

### 老师预检与待提交

老师在临时目录运行以上完整修改，原训练与验证断言、五组参数无损往返、同输入预测与固定验证损失检查通过；另外用独立 Python 子进程加载文件并前向，确认可脱离旧模型对象使用，并检查错误结构格式或错误参数形状会被拒绝。未在仓库保存老师预检权重。参考末尾输出（路径按实际实操位置显示）：

```text
权重保存路径： D:\projects\playatry\model\learn_rnn_step005.npz
权重文件字节数： 2686396
加载后训练步数： 5
加载前后验证损失： 8.628433227539062 8.628433227539062
模型保存、重载与预测一致性检查：通过
```

运行 python scripts/learn/11_train_steps.py，提交末尾保存与重载输出。回答：一、为什么除了五组权重，还要保存词表的编号顺序？二、加载模型时构造函数使用 seed=0，而训练初始 seed=42，为什么完整覆盖五组参数后预测仍应相同？完成后复查脚本与权重，补充笔记、仓库例外与 CI 保存重载说明，并完整同步 GitHub。

### 第二十六课实操复查：通过，词表与随机初值概念继续讲解

老师检查学习者实际代码并重新运行，全部训练、验证及保存重载断言通过。实际文件 model/learn_rnn_step005.npz 大小为 2686396 字节，加载步数为 5；加载前后固定验证子集损失均为 8.628433227539062。五组参数逐元素完全一致，预测分数在容差内一致。老师另在新 Python 进程直接加载此文件，未执行训练或数据统计，确认参数数 719723、词表大小 5547、隐藏维度 64，输入“春风”可得到有限的 (1,2,5547) 分数，保存词表与当前固定词表完全一致。

仅清理脚本行尾空白，前后语法树一致，未修改学习者计算逻辑。为履行本课完整上传约定，给该学习里程碑添加明确 .gitignore 例外，更新 model/README.md 与 CI 说明。权重 SHA-256 为 db80ce4b3d2bc584f7e746b456223e5084c731c45d31b35fc832e6eca3a9c45f。这里只训练五步，不将保存成功解释为生成质量已合格。

两道口述题学习者回答“我不知道”，因此记录为代码实操通过、概念继续解释，不推进下一课。

第一题：模型参数按编号工作，并不知道某个编号天然代表哪个汉字。假设训练词表编号 1 对应“春”、2 对应“秋”（仅为示意，并非项目实际编号），加载时却交换顺序，那么表示表第一行会被当作“秋”的向量，输出第一列也会被解释成“秋”的分数。即使权重数值一点没变，输入编码与输出解码仍会错误。因此必须恢复训练时完全相同的字符到编号映射，不能只检查字符集合或词表大小相同。

第二题：seed 只决定构造模型时生成的临时随机初值；load_checkpoint 随后逐组执行 model.params[name]=value.copy()，把这五组初值全部替换成文件保存的训练参数。最终预测使用文件中的值，而不是 seed=0 生成的值。在相同词表、模型计算和输入下，参数相同就应得到一致预测。不能将“完整加载后不依赖临时初始化种子”推广成“训练初始种子不会影响训练过程”。

下一次只确认词表映射：假设训练时编号 1 代表“春”，加载后改成代表“秋”，即使权重完全不变，编号 1 的含义是否仍然正确？等待学习者回答，不提前标记全部概念掌握。

### 词表映射复核纠错：权重相同不等于字符含义相同

学习者回答“正确”，此处应为“不正确”，说明仍需区分数字计算与字符解释。沿用示意编号：训练时 tokens[1]="春"，模型在某处给编号 1 较高分数，按原词表应解释为“春”；若加载后改为 tokens[1]="秋"，同一个输出编号就会被错误解释为“秋”。权重没变只能保证相同编号输入下的数字计算不变，不能保证编号对应的汉字含义不变。真实文字输入若也用错误词表编码，还会送入不同的编号、选错表示表行。

因此加载时必须同时恢复训练时的参数和词表编号顺序。上述编号只为教学示意，并非项目实际词表编号。本次不推进下一课，不修改练习代码；第二十六课实操已通过，词表映射概念继续复核。

下一次只问直接解码：如果 tokens[1] 被改成“秋”，模型输出编号 1 时，程序实际显示“春”还是“秋”？待学习者回答。

### 直接解码复核：回答正确

学习者回答“秋”，正确：解码查的是当前词表 tokens[1]，它若被改为“秋”，同一个输出编号 1 就会显示“秋”。这确认了词表顺序改变会改变编号的字符解释，保存权重时必须保留配套的原词表。本次不将该单项回答扩展为已掌握全部保存加载原理。

词表问题已通过直接例子巩固，继续确认加载参数来源：load_checkpoint 创建模型后，执行 model.params[name]=value.copy() 覆盖五组参数；最终预测使用的是临时随机初值，还是从文件读出的训练参数？该问题待回答，本次不新增练习代码。

### 加载参数来源复核：回答正确

学习者回答“文件里保存的训练参数”，正确。加载函数完整覆盖五组临时随机参数后，预测使用文件中保存的数值，所以临时构造模型时 seed=0 与原训练初始 seed=42 不同，也不影响恢复后的预测；前提是模型结构、词表顺序与输入保持一致。初始训练种子仍可能影响从头训练的结果，不能混为一谈。

第二十六课保存与重载实操、参数及预测一致性检查通过；词表解码和加载参数来源经补充讲解后的口述复核均回答正确。模型权重已作为学习里程碑纳入仓库。本次仅补记回答，未修改代码或重新训练。下一课可独立加载已保存模型，实现逐字生成；该生成练习尚未实施。

## 第二十七课：独立加载权重，给定开头逐字生成

状态：老师已在内存中预检以下完整代码，待学习者新建 scripts/learn/12_generate_text.py 并亲自输入、运行和回答。未替学习者创建脚本，未重新训练或修改已保存权重。本课实现最简单的贪心生成，不使用随机采样、温度或 top-k。

### 生成循环与数学

输入“春风”，将它编码后前向计算，取最后一个输入位置的分数向量，选择分数最大的编号 i*=argmax_i z_i，再把该编号接到输入末尾。下一步前向包含刚生成的编号，重复四十次。完整循环是：当前文字编号→前向→取最后位置分数→选择一个编号→追加编号→下一轮。

训练时各位置使用真实前一字符作为输入；生成时开头之后使用自己选择的字符继续预测，错误和重复也可能被继续输入。所有参数固定，不计算梯度、不调用更新。模型输出 (1,T,V)，logits[0,-1,:] 是最后位置预测下一个字符的候选分数，之前位置的输出对应更早的预测位置，不是本次应追加的字符。

因为 softmax 对各分数使用同一个正分母，指数函数严格递增，所以 argmax softmax(z)=argmax z；贪心选择无需先求概率。argmax 返回编号而不是字符，要用保存词表 tokens[next_id] 解码。分数恰好相同时，NumPy argmax 选择第一个最大值；本课同参数同开头的生成是确定性的。

### 第一块：新文件导入

```python
from importlib import import_module  # 按字符串导入已有模型和权重加载函数。
from pathlib import Path  # 根据脚本位置找到已保存的权重文件。
import numpy as np  # 创建输入编号数组，并选择分数最高的候选。
```

### 第二块：接着添加生成函数

```python
def generate(model, vocab, prompt, new_tokens=40, context_size=32):  # 根据开头逐步选择新编号，不更新任何参数。
    assert isinstance(prompt, str) and len(prompt) > 0, "开头文字不能为空"  # 至少提供一个字符，模型才能预测后续。
    assert isinstance(new_tokens, int) and new_tokens >= 0 and isinstance(context_size, int) and context_size > 0, "生成步数或上下文长度不合法"  # 允许生成零步，但上下文长度必须为正。
    tokens = vocab["tokens"]  # 使用权重文件内保存的词表顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 根据保存词表恢复字符到编号的映射。
    token_ids = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 未见字符使用已有未知编号，不扩充词表。
    generated_ids = []  # 只记录新生成的编号，便于检查生成步数。
    for _ in range(new_tokens):  # 每次循环选出一个新编号。
        x = np.asarray([token_ids[-context_size:]], dtype=np.int64)  # 只取最近的上下文，整理成一条二维输入。
        logits = model.forward(x)  # 只做前向，得到 (1,上下文长度,词表大小) 的分数。
        next_id = int(np.argmax(logits[0, -1, :]))  # 最后位置预测下一个编号，贪心选择分数最高的一项。
        token_ids.append(next_id)  # 将本次结果加入输入，下一轮就能读到它。
        generated_ids.append(next_id)  # 单独保存本次生成的编号。
    text = prompt + "".join(tokens[index] for index in generated_ids)  # 保留原始开头，将新编号按保存词表解码。
    return text, generated_ids  # 返回文字和新编号，特殊未知符号也按词表显示。
```

本课每步重新前向，最多取最近 32 个编号，并按现有 forward_hidden 从零初态重算这段上下文。这样是一个明确的生成策略，不表示 RNN 结构只能处理 32 字；超过 32 的更早历史不会参与本步计算。以后讨论缓存隐藏状态时，需区分完整历史递推与滑动窗口重算，两者并不自动等价。

未见的开头字符映射到文件中保存的 unk_id，不更改词表；输出文字保留原始 prompt 仅为显示，模型实际上无法区分映射到同一未知编号的不同字符。若模型生成 <UNK>，输出会显示这个特殊字符串；因此应检查生成编号数为 40，不能普遍断言可见字符串恰好新增 40 个 Unicode 字符。本课不额外屏蔽未知编号或强制停止符；生成步数就是循环次数。

### 第三块：添加独立加载与检查入口

```python
def main():  # 独立加载权重并生成文字，不运行训练脚本的 main。
    network = import_module("07_tiny_rnn")  # 取得前向模型实现。
    checkpoints = import_module("11_train_steps")  # 只复用加载函数，模块的训练入口不会被执行。
    root = Path(__file__).resolve().parents[2]  # 从当前脚本位置找到项目根目录。
    checkpoint_path = root / "model" / "learn_rnn_step005.npz"  # 使用上一课已经保存并上传的学习权重。
    model, vocab, completed_steps = checkpoints.load_checkpoint(checkpoint_path, network)  # 从文件恢复参数和配套词表。
    prompt = "春风"  # 给定希望模型继续写下去的开头。
    saved_params = {name: param.copy() for name, param in model.params.items()}  # 保存副本，检查生成过程没有修改权重。
    text, generated_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32)  # 连续选择四十个新编号。
    repeat_text, repeat_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32)  # 同参数同输入再次贪心生成，核对确定性。
    assert len(generated_ids) == 40 and all(0 <= index < len(vocab["tokens"]) for index in generated_ids), "生成编号数量或范围不正确"  # 每步恰好生成一个合法编号。
    assert text.startswith(prompt) and text == repeat_text and generated_ids == repeat_ids, "开头或重复生成结果不一致"  # 贪心生成没有随机抽样。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "生成过程修改了参数"  # 生成只前向，不训练。
    print("加载权重的训练步数：", completed_steps)  # 当前权重只训练了五步。
    print("输入开头：", repr(prompt))  # 清楚显示开头中的特殊字符。
    print("新生成编号数：", len(generated_ids))  # 应为 40，不包含开头编号。
    print("生成结果：", repr(text))  # 用 repr 保留换行等特殊字符，方便核对输出。
    print("权重独立加载、贪心生成与参数不变检查：通过")  # 正确性检查通过不代表文字质量已经良好。


if __name__ == "__main__":  # 直接运行新文件时才开始生成演示。
    main()  # 加载磁盘权重并生成，不执行训练更新。
```

导入 11_train_steps 只定义函数，不执行受 main 保护的训练入口；因此本脚本不读取训练批次、不重建词表、不重新训练，不会出现第六课的数据统计输出。完整词表和参数都来自上一课权重文件。保存参数副本是教学检查，核对生成过程中参数保持不变；第二次生成检查确定性，不是再训练一次。

### 如何看待结果与效率

当前权重只训练五步，尚未完成一个 epoch，更没有完成所有书籍的充分训练。输出可能重复、罕见字符较多或不通顺。本课通过标准是模型可从文件独立加载、逐步生成合法编号并保留参数，不把生成质量当作代码正确性断言。下一阶段再学习不同解码方式与更充分的训练，不能用随机抽样掩盖模型尚未学好的事实。

设生成步数 G、最大上下文 K，每步复用现有 forward，会重算最长 K 步状态和全部位置输出，主要计算量约 O(G*K*(H²+HV))，另有每步 argmax 的 O(V) 开销。当前 V=5547、H=64、K=32，单份最大 logits 有 32*5547 个 float32，数据区为 710016 字节，不包含参数与其他数组。这种写法优先清楚，后面可学习只计算最后位置输出和复用隐藏状态来减少重复工作；本课没有声称已做性能优化。

### 老师预检与待提交

老师在内存中组合以上完整新脚本，独立加载已保存权重，不执行训练，全部断言通过。另用可记录输入的小模型检查新编号正确反馈、上下文截断、生成零步及未知开头字符映射。未修改仓库代码或权重。以下是当前文件和环境下的参考输出，跨平台极接近的候选分数若顺序变化，文字可能不同，以流程断言为准：

```text
加载权重的训练步数： 5
输入开头： '春风'
新生成编号数： 40
生成结果： '春风忏帐程踊昏膻帑幔徐抬咯诐烟掇窃旭崇艮狠仄揿箔刘踏染攒狭诰幔凿傍匡媛箕馄筒纬样州比'
权重独立加载、贪心生成与参数不变检查：通过
```

运行 python scripts/learn/12_generate_text.py，贴出完整输出。回答：一、为什么选择 logits[0,-1,:]，而不是 logits[0,0,:]？二、选出一个新编号后，为什么要 token_ids.append(next_id)？完成后复查实际脚本、记录输出与回答、加入 CI 并同步 GitHub。若文字重复先如实记录，不据此要求随意调整学习率或认定保存加载错误。

### 第二十七课实操复查：通过，生成位置与反馈继续讲解

老师检查学习者实际新建脚本并重新运行，全部断言通过。独立加载已训练五步的权重，输入“春风”，新增 40 个编号，实际输出为“春风忏帐程踊昏膻帑幔徐抬咯诐烟掇窃旭崇艮狠仄揿箔刘踏染攒狭诰幔凿傍匡媛箕馄筒纬样州比”，与预检一致。没有重新执行训练或数据统计，同参数同输入的重复生成一致，五组参数未变。该输出尚不通顺，仅说明生成流程走通，不代表语言质量合格。

仅清理脚本行尾空白，清理前后语法树一致，未修改学习者实现。新脚本已加入 GitHub Actions 已完成课程检查。

两道口述题学习者回答“我不知道”，记录为实操通过、概念继续讲解，不推进下一课。

第一题拆开解释：输入“春风”时 logits 形状为 (1,2,5547)。logits[0,0,:] 对应只读到“春”后，预测“春”之后的字符；logits[0,1,:] 对应读完“春风”后，预测整个开头之后的字符。Python 的 -1 表示最后一个位置，所以选 logits[0,-1,:] 才是在续写当前输入末尾。这里解释的是每个位置的预测任务，不是说尚未训练好的模型必然能正确猜到真实下一字。

第二题：本次实际首先选择“忏”的编号，将它追加后，下一轮输入从“春风”变为“春风忏”，后续才会基于增长后的上下文继续预测。token_ids 是模型下一轮实际使用的编号历史，generated_ids 是本次新增编号的记录；前者影响后续预测，后者用于最终解码与计数。若只给 generated_ids 追加、不更新 token_ids，每次前向仍读取同一开头，确定性的贪心策略就会重复同一个结果。append 不改变模型权重，只改变下一轮输入。

下一次只确认第一个概念：要在“春风”末尾继续写，应使用只读到“春”时的分数，还是读完“春风”时的分数？等待学习者回答，不提前标记概念掌握。

### 补充问答：老师为什么知道输出，能否直接看懂权重

学习者追问老师为什么知道预测结果，是否能通过权重看懂预测值，以及如何理解权重。需要明确证据来源：老师实际运行了学习者脚本，观察到生成结果，随后加载同一权重进行前向计算并检查候选分数；并非凭肉眼查看某个权重元素就识别出预测汉字。

再次实际检查 model/learn_rnn_step005.npz：输入“春风”编码为 [[2086,5263]]，状态形状 (1,2,64)，分数形状 (1,2,5547)。最后位置前三个候选为：编号 1517“忏”，分数 0.5957961678504944，softmax 概率 0.00032414557063020766；编号 1868“授”，分数 0.545968770980835，概率 0.00030839000828564167；编号 167“伞”，分数 0.5351401567459106，概率 0.00030506859184242785。本课 greedy 的 argmax 因此选 1517，查保存词表得到“忏”。概率仅为说明而额外计算，原生成代码不必先计算 softmax。

可解释的是计算结构：按编号从 E 查行，依次计算 h_t=tanh(E[id_t]+h_(t-1)@R+b)，再算 z=h_last@O+b_o；z_i 是当前上下文下候选编号 i 的分数，O 的第 i 列参与该候选的评分。输入变化会改变隐藏状态，同一份输出权重也就给出不同评分。

权重是数值运算中的系数，词表定义编号的字符含义；每个隐藏分量或某个浮点权重通常没有预先规定的独立中文语义，不能仅凭数值就命名为“春天”“人物”等。若要研究某些内部方向是否具有可解释作用，需要另做输入对比、相近向量分析、消融等实验，不能把猜测当成已经验证的结论。本次没有进行这些语义解释实验。

也应区分当前模型的预测与自然语言合理性：这里只训练五步，最高候选概率约为 0.0324%，生成文字尚不通顺。检查到“忏”分数最高，只能说明当前模型与选择规则为何输出它，不能证明它是语言上合理的续写。本次仅读取权重、前向和记录笔记，未改变模型或训练代码；此前的位置选择与编号反馈口述题仍待学习者回答。

### 第二十七课位置选择口述复核与课程推进

学习者回答要在“春风”末尾继续写，应看“读完春风时”的分数，正确：使用最后位置 logits[0,-1,:]。学习者明确要求没有问题就进入下一课，按要求推进。追加编号作为下一轮输入的作用已讲解，并在后续生成循环中继续巩固；不虚构第二题已有独立回答。本课实操通过，当前继续学习解码选择。

## 第二十八课：概率采样与温度

状态：老师已在内存中预检以下修改，待学习者亲自修改 scripts/learn/12_generate_text.py 并运行。保留上一课贪心默认行为，新增按概率采样的可选分支；未修改权重、未重新训练或替学习者修改实际脚本。

### 从固定最高分到按概率抽取

贪心总取 argmax；概率采样先将分数转换为分布，再根据各候选概率抽取。假设三个候选概率是 [0.6,0.3,0.1]：贪心总选第零项，而采样时三项都有机会；概率表示重复抽样的统计规律，不能要求每抽十次恰好得到 6、3、1 次。

温度 T>0 时 p_i(T)=exp(z_i/T)/Σ_j exp(z_j/T)。T=1 使用模型原来的 softmax 分布；T<1 放大分数差距，分布更集中；T>1 缩小差距，分布更平缓。所有温度都在同一组 logits 上操作，不修改模型权重，不进行训练。低温并不保证每次都选最大项，高温也不保证文章更好。

对于非均匀示例 p=[0.6,0.3,0.1]，取 z=ln(p)：T=0.5 时，概率相当于原概率平方后归一化，[0.36,0.09,0.01]/0.46；T=2 时，相当于开平方后归一化。原分数相等时，无论温度多少仍为均匀分布。

### 第一块：在 generate 前顶格添加概率函数

```python
def sampling_probabilities(scores, temperature):  # 用温度把最后位置的分数转换成抽样概率。
    scores = np.asarray(scores, dtype=np.float64)  # 用双精度计算概率，供随机抽样使用。
    assert scores.ndim == 1 and scores.size > 0 and np.isfinite(scores).all(), "需要有限的一维候选分数"  # 每次只处理最后一个位置。
    assert np.isfinite(temperature) and temperature > 0, "采样温度必须是有限正数"  # 温度参与除法，不能为零或负数。
    shifted = (scores - scores.max()) / temperature  # 先统一平移，再按温度缩放分数差距。
    weights = np.exp(shifted)  # 将缩放后的分数转换为非负权重。
    return weights / weights.sum()  # 除以权重总和，得到概率总和为 1 的分布。
```

先减去最大值保证最大指数为 1，再除以温度；对全部候选统一平移不改变 softmax。双精度概率供 rng.choice 使用，模型参数仍是 float32。本课使用普通正温度，不把 T=0 当作合法除数；想贪心时使用下一块的 temperature=None。

### 第二块：只修改 generate 的三处，其余原代码保留

把函数定义整行替换为：

```python
def generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=None, seed=42):  # 默认贪心，指定正温度时按概率采样。
```

在原 for _ in range(new_tokens) 前、generated_ids=[] 后添加：

```python
    rng = np.random.default_rng(seed)  # 每次生成创建一个局部随机数生成器，在整个生成循环中持续使用。
```

把原 next_id=int(np.argmax(logits[0,-1,:])) 那一整行替换为下面六行，保持在 for 循环内的八格缩进：

```python
        scores = logits[0, -1, :]  # 读完当前上下文后，用最后位置的分数预测下一字。
        if temperature is None:  # 未指定采样温度时保留上一课的贪心方式。
            next_id = int(np.argmax(scores))  # 直接选择分数最高的编号。
        else:  # 指定正温度后，允许按概率选择其他候选。
            probabilities = sampling_probabilities(scores, temperature)  # 温度调整概率的集中程度，不修改模型参数。
            next_id = int(rng.choice(len(tokens), p=probabilities))  # 根据每个候选的概率抽取一个合法编号。
```

原 token_ids.append(next_id)、generated_ids.append(next_id)、返回文字等代码保留。无论选择哪个分支，都必须将选中的编号加入下一轮上下文。rng 创建在生成循环外，让抽样状态每步继续向前；每次 generate 调用用给定 seed 新建局部随机数生成器。

两个“种子”作用不同：权重加载时的临时初始化种子随后被文件参数覆盖；这里 seed 控制生成时抽样序列，不改变权重。同一参数、词表、开头、温度、上下文策略、采样种子和一致环境下结果可复现；不同种子通常产生不同结果，但并非数学上保证每次不同。

temperature=None 保留旧贪心行为，所以原 main 中的贪心结果与断言仍应通过。temperature=1.0 使用原概率采样；T=1 不表示每次都选择最高分。不能把采样种子当成额外训练参数。

### 第三块：在原 main 末尾追加温度与采样检查

放在原 print("权重独立加载、贪心生成与参数不变检查：通过") 后，仍为 main 内四格缩进，入口判断仍顶格。

```python
    demo_scores = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float64))  # 构造温度为 1 时恰好得到这三个概率的分数。
    for temperature in (0.5, 1.0, 2.0):  # 比较较低、标准和较高温度。
        probabilities = sampling_probabilities(demo_scores, temperature)  # 对同一组分数只改变温度。
        assert np.isfinite(probabilities).all() and np.all(probabilities >= 0) and np.isclose(probabilities.sum(), 1.0), "采样概率不合法"  # 核对抽样分布。
        print(f"温度 {temperature} 的演示概率：", np.round(probabilities, 6))  # 只为展示而四舍五入，抽样不使用截短概率。
    sampled_text, sampled_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 用保存权重按原始概率连续抽样四十步。
    repeated_sample, repeated_ids = generate(model, vocab, prompt, new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 同参数、温度、开头和采样种子再次生成。
    assert sampled_text == repeated_sample and sampled_ids == repeated_ids, "固定采样种子没有复现结果"  # 局部随机数状态应从同一种子出发。
    assert len(sampled_ids) == 40 and all(0 <= index < len(vocab["tokens"]) for index in sampled_ids), "采样编号数量或范围错误"  # 每一步必须得到一个合法编号。
    assert all(np.array_equal(param, saved_params[name]) for name, param in model.params.items()), "采样生成修改了参数"  # 改变解码策略不改变权重。
    print("温度 1.0、采样种子 42 的结果：", repr(sampled_text))  # 观察概率抽样与贪心结果的区别。
    print("温度概率、固定种子采样与参数不变检查：通过")  # 检查流程正确，不判断文字质量。
```

打印时 round 只影响展示，不用四舍五入后的数值抽样，以免概率和偏离 1。示例分布只有三项便于手算，真实模型抽样仍在全部 5547 个候选上进行。本课尚未实现 top-k、top-p 或未知编号屏蔽。

### 生成质量与效率

当前权重只训练五步，概率分布很分散，采样输出仍可能不通顺或出现罕见字符；采样改变选择方式，不能替代训练或保证生成质量改善。本课先验证解码算法，下一阶段再推进训练与评估。

主要前向计算量与上一课相同；每次抽样额外对 V 个候选计算概率并选择，约 O(V) 级开销，临时概率向量为 V 个 float64，当前约 44376 字节，不增加模型参数。没有做训练速度或生成质量基准比较。

### 老师预检与待提交

老师在内存中组合修改运行，原贪心断言与新采样断言均通过；另检查大分数平移稳定性、相等分数的均匀分布、低温示例的精确分布与非法温度拒绝。没有修改学习者脚本或权重。新增参考输出如下，采样文字在不同 NumPy 实现或浮点环境下可能有差异，以流程与复现断言为准：

```text
温度 0.5 的演示概率： [0.782609 0.195652 0.021739]
温度 1.0 的演示概率： [0.6 0.3 0.1]
温度 2.0 的演示概率： [0.472734 0.334273 0.192993]
温度 1.0、采样种子 42 的结果： '春风衽殃返腭匹鲂蜩觌哏沃旱隶索赀毫屎畚兹赡篡蜓撼髽钢裩娈浙俦园肚蕴骥挛旺浴妯哭渍层翕'
温度概率、固定种子采样与参数不变检查：通过
```

运行 python scripts/learn/12_generate_text.py，提交新增概率与采样输出。回答：一、概率为 [0.6,0.3,0.1]，采样是否每次都选第零项？二、温度从 1 降到 0.5，这个非均匀示例的概率分布会更集中还是更平缓？完成后复查实际代码、补记结果、更新 CI 说明并同步 GitHub。

### 第二十八课实操纠错：恢复循环并移除重复分支

学习者运行报第 21 行 IndentationError: unexpected indent。老师检查实际文件发现，rng 初始化后直接出现八格缩进的 scores 与选择分支，没有有效的外层循环；原 for _ in range(new_tokens) 被粘在上一行 # 注释后面，整段成为注释，不再定义循环。此外循环前多粘贴了一份选字分支，在 logits 尚未计算时就引用它。

老师按检查纠错约定直接修复：删除 rng 后多余的一份 scores/if/else 分支，恢复独立的四格缩进 for 行，保留原循环内 x→forward→scores→贪心或采样→两次 append 的顺序。缩进规则：函数体四格；for 循环体八格；循环内 if/else 的分支语句十二格。rng 在循环外只初始化一次。已保留学习者用于对照的旧 argmax 注释，另清理行尾空白。

修复后语法编译与完整运行通过；旧贪心输出保持一致，温度示例 0.5/1/2 分别为 [0.782609,0.195652,0.021739]、[0.6,0.3,0.1]、[0.472734,0.334273,0.192993]。温度 1、采样种子 42 的输出为“春风衽殃返腭匹鲂蜩觌哏沃旱隶索赀毫屎畚兹赡篡蜓撼髽钢裩娈浙俦园肚蕴骥挛旺浴妯哭渍层翕”。概率、重复采样、合法编号和参数不变断言均通过，没有重新训练或修改权重。

本课 CI 说明已更新为覆盖温度与采样。本次记录为老师修复后的实操检查通过，口述两题尚未回答，不提前标记概念掌握，也不推进下一课。

### 学习者重新运行确认：通过

缩进修复后，学习者再次实际运行脚本，提交完整输出。原贪心输出、新生成编号数 40、温度 0.5/1/2 的三组演示概率，以及温度 1、采样种子 42 的文字均与老师检查一致；贪心确定性、固定种子采样和参数不变检查通过。本次没有新的代码修改，不重复执行无变化的检查。

第二十八课实操已由学习者重新运行确认，口述仍待回答：概率 [0.6,0.3,0.1] 的采样是否每次都选择编号 0；该非均匀示例温度从 1 降至 0.5 后是更集中还是更平缓。当前权重仍只训练五步，文字质量未因流程检查通过而被认定合格，不推进下一课。

### 第二十八课口述复核：两题均正确

学习者回答第一题“否”、第二题“更集中”，均正确。概率采样时编号 0 的概率虽最高，其他非零概率候选仍可能被选中；温度从 1 降到 0.5 后，这个示例第一项概率从 0.6 升至约 0.7826，分布更集中，但仍不等于每次必选最大项。

第二十八课实操与问答通过。本次仅补记回答，未修改代码、权重或重新训练。当前模型仍只训练五步，下一阶段可增加实际训练步数，并使用固定验证子集与生成结果共同观察模型变化；具体训练配置和记录将在下一课实现，不将本次采样检查视为模型语言能力已经合格。

## 第二十九课：增加训练步数，对照验证损失与生成结果

状态：老师已在临时目录完成 200 步预跑，待学习者新建 scripts/learn/13_training_experiment.py 并手敲运行。未替学习者创建脚本或在仓库生成新权重。原 model/learn_rnn_step005.npz、11 和 12 脚本保持不变；本课新模型从种子 42 初始化，在同一次训练轨迹的 0、5、50、200 步做对照。

### 实验控制与范围

固定模型结构 V=5547、H=64，B=16、T=32、SGD 学习率 0.1、全局裁剪阈值 1、模型种子 42。四个检查点都评估同一个验证集顺序前三批（1536 个预测位置），并用同一“春风”开头生成 40 个编号；采样温度 1、采样种子 42、生成上下文上限 32 始终相同。验证保持参数不变，测试集不使用。

这次固定 random.Random(42) 打乱训练章节列表，使短训练接触四本书，章内字符顺序与章节边界不变。旧五步权重采用原顺序训练，本实验采用新顺序，所以不能把旧文件与新 200 步结果的差异全归因于步数；主要比较本实验同一顺序轨迹上的 0、5、50、200 步。本次只打乱一次，不开始新 epoch。

模型只创建一次，训练参数递推继续使用 θ_k=θ_(k-1)-η*clip(∇L_(batch_k)(θ_(k-1)))。每个检查点先固定参数评估，再生成样例和保存文件，不用验证梯度更新参数。训练日志是各批更新前损失，不与最终模型固定验证损失混为同一个指标。

### 第一块：新文件导入

```python
from importlib import import_module  # 复用已有数据、模型、评估、保存与生成代码。
from itertools import islice  # 只取本次实验需要的前 200 个训练批次。
import json  # 将实验配置、验证损失和生成文字保存成可复查的记录。
import random  # 用固定种子打乱章节顺序，不打乱章内字符。
import numpy as np  # 检查损失和参数是否为有限数值。

network = import_module("07_tiny_rnn")  # 导入模型与梯度裁剪，不执行旧模型演示。
training = import_module("11_train_steps")  # 复用固定参数评估和权重保存，不执行旧训练入口。
decoding = import_module("12_generate_text")  # 复用贪心与采样生成，不执行旧生成演示。
```

三个已有模块都有 main 保护，导入不会执行旧训练或生成。第六课数据模块在 main 中导入，仍会先显示旧数据检查。

### 第二块：记录一个检查点

```python
def inspect_checkpoint(model, vocab, validation_batches, step, output_dir):  # 对当前参数做固定评估、生成和保存。
    validation_loss, positions = training.evaluate(model, validation_batches, network)  # 始终评估同一份固定验证子集，不更新参数。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 固定开头和生成长度，观察贪心结果。
    sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定温度和抽样种子，减少比较中的随机干扰。
    checkpoint_path = output_dir / f"lesson29_step{step:03d}.npz"  # 各步使用不同文件名，不覆盖原五步学习权重。
    training.save_checkpoint(checkpoint_path, model, vocab, step)  # 保存当前参数，以后可以独立重载这个检查点。
    print(f"检查点 step={step} 固定验证损失={validation_loss:.6f}")  # 显示同一子集上的模型表现。
    print("贪心：", repr(greedy))  # 如实保留重复或不通顺的输出。
    print("采样：", repr(sampled))  # 同步记录概率采样结果，不凭一条样例判断总体质量。
    return {"step": step, "validation_loss": validation_loss, "validation_positions": positions, "greedy": greedy, "sampled": sampled, "checkpoint": checkpoint_path.name}  # 返回可写入 JSON 的本次观察。
```

输出目录中的文件名分别为 lesson29_step000.npz、lesson29_step005.npz、lesson29_step050.npz、lesson29_step200.npz。step000 是未训练基线；后面各文件是同一次训练的参数快照，不是另外初始化重训。相同路径再次运行实验会被本次结果覆盖，原 learn_rnn_step005.npz 不受影响。评估和生成不训练模型，保存的是检查点当前参数。

### 第三块：main 中准备固定数据与模型

```python
def main():  # 从头训练一个模型，比较不同训练步数的检查点。
    data = import_module("06_make_windows")  # 读取训练集和固定词表，首次导入仍会显示旧数据检查。
    batch_size = 16  # 每个完整训练批次使用 16 个窗口。
    seq_length = 32  # 窗口长度和步长均为 32。
    max_steps = 200  # 本次先连续更新 200 步，仍未完成整个训练集的一轮。
    learning_rate = 0.1  # 保持普通梯度下降学习率不变。
    max_norm = 1.0  # 保持全局梯度裁剪阈值不变。
    train_records = list(data.train_records)  # 复制章节列表，不改变原数据模块的章节顺序。
    random.Random(42).shuffle(train_records)  # 固定随机打乱章节顺序，章内字序保持原样。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取独立验证集。
        validation_records = [json.loads(line) for line in file]  # 按原顺序保留验证章节。
    validation_batches = list(islice(data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size), 3))  # 沿用之前固定的前三个验证批次。
    assert len(validation_batches) == 3, "固定验证子集不足三批"  # 确认本次评估范围与之前课程一致。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 只创建一次模型，之后在同一训练轨迹上比较步数。
    output_dir = data.ROOT / "model"  # 将四个独立检查点写入模型目录。
    history = [inspect_checkpoint(model, data.vocab, validation_batches, 0, output_dir)]  # 更新之前先记录零步基线。
    chapter_to_book = {record["id"]: record["book"] for record in train_records}  # 根据章节标识追踪训练窗口来自哪本书。
    book_windows = {record["book"]: 0 for record in train_records}  # 为四本书分别累计实际参与训练的窗口数。
    batches = data.iter_batches(data.iter_windows(train_records, seq_length, seq_length), batch_size)  # 按本次固定章节顺序提供连续训练批次。
    completed_steps = 0  # 累计实际完成的更新次数。
```

train_records 是原章节列表的浅拷贝，只改变这个列表的排列，不改文本或磁盘数据。打乱单位是完整章节，不是字，不是把下一字目标随机打散。验证子集仍沿用先前课程的相同顺序前三批；它不代表四本书全部验证章节的表现。

### 第四块：接在 main 内连续训练

```python
    for step, batch in enumerate(islice(batches, max_steps), start=1):  # 连续处理前 200 个批次，每批更新一次。
        x, y = batch["x"], batch["y"]  # 取真实输入和下一字目标编号。
        loss, raw_grads = model.loss_and_grad(x, y)  # 用当前参数完成本批前向与完整反向。
        assert np.isfinite(loss), "训练损失异常"  # 不让无效损失继续进入训练记录。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 汇总全部梯度后，再按全局范数裁剪。
        for name in model.params:  # 使用本批完整梯度更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 下一批继续使用本次更新后的参数。
        assert all(np.isfinite(param).all() for param in model.params.values()), "模型参数异常"  # 发现无效参数时停止。
        for chapter_id in batch["chapter_ids"]:  # 按实际窗口来源统计四本书的训练覆盖。
            book_windows[chapter_to_book[chapter_id]] += 1  # 本批每行窗口为对应书籍增加一次计数。
        completed_steps = step  # 记录已经完成的更新步数。
        if step % 25 == 0:  # 每隔 25 步显示一次进度，不逐步刷屏。
            print(f"训练 step={step} 更新前批次损失={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f}")  # 不要求不同批次损失单调下降。
        if step in (5, 50, 200):  # 在预先选定的步数评估同一个模型的当前参数。
            history.append(inspect_checkpoint(model, data.vocab, validation_batches, step, output_dir))  # 记录固定验证损失和相同设置下的生成结果。
```

enumerate(...,start=1) 提供一开始的更新步数；每个批次先反向、裁剪、完成五组参数更新，再把 step 记入计数并按计划检查。每隔 25 步打印一行进度，固定 5、50、200 步才评估、生成、保存。不要求损失随每条训练日志单调下降；训练检查参数有限，验证变化和文字质量如实观察。

### 第五块：接在 main 内保存实验记录，再添加入口

```python
    assert completed_steps == max_steps and all(count > 0 for count in book_windows.values()), "本次实验步数或四书覆盖不符合预期"  # 当前数据和固定顺序应完成 200 步并接触四本书。
    report = {  # 汇总本次实验的配置、数据来源和观察结果。
        "config": {  # 固定比较条件，方便以后复现实验。
            "steps": max_steps, "batch_size": batch_size, "seq_length": seq_length,  # 记录训练量与批次尺寸。
            "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42, "chapter_shuffle_seed": 42,  # 记录更新规则与初始化顺序。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42, "validation_batches": 3,  # 记录固定生成和验证条件。
        },  # 配置记录结束。
        "train_sha256": data.vocab["train_sha256"],  # 记录固定词表对应的训练文件指纹。
        "book_windows": book_windows,  # 记录四本书实际提供的训练窗口数。
        "history": history,  # 保存四个检查点的验证损失和生成样例。
    }  # 完整实验记录组装结束。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson29.json"  # 为本次实验保留独立的可复查记录。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 创建记录目录，已经存在时继续使用。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写入中文可读的 JSON，不覆盖学习笔记。
    print("四本书实际训练窗口数：", book_windows)  # 确认四本书都实际提供了本次训练数据。
    print("完成更新步数：", completed_steps)  # 应为 200，而不是 200 个 epoch。
    print("实验记录路径：", report_path)  # 显示配置、验证损失和生成样例的保存位置。
    print("多步训练、固定验证与生成对照记录：完成")  # 只说明流程完成，不保证所有模型指标都更好。


if __name__ == "__main__":  # 直接运行脚本时才开始本次训练实验。
    main()  # 训练并记录零步、五步、五十步和两百步的模型。
```

book_windows 统计实际训练窗口，避免误以为“加载了四本书”就等于“本次四本书都参与更新”。当前数据和固定种子下四本书均会被访问，但窗口数不相等；章节级打乱不是书籍等量采样。200 步共 3200 个窗口、102400 个预测位置，约为 4787 步一轮的 4.18%，仍未完成一个 epoch。上述覆盖断言针对本课固定配置，不保证把 max_steps 改得很小后仍成立。

JSON 记录保存全部对照条件、数据指纹、验证损失和原样文字，不能只留“看起来最好”的输出。这里没有自动选择最优检查点，也不声称检查点验证损失必然依次降低；若升高也保留。保存四个独立小权重是为了之后独立复查。

### 老师预跑观察

固定验证子集的平均交叉熵：

| 已完成 step | 验证损失 |
|---:|---:|
| 0 | 8.651848 |
| 5 | 8.629895 |
| 50 | 8.174481 |
| 200 | 7.676629 |

本环境预跑完整实验（含数据导入检查、四次验证与生成、保存文件）约 18.1 秒，实际实操速度取决于 CPU 与 NumPy，不保证相同用时。模型参数始终为 719723 个，训练步数增加主要增加时间；四个权重文件各约 2.7 MB，合计约 10.7 MB，不增加单个模型的参数数量，也不需要在内存中同时保留四个模型。

观察重点：验证损失下降表明在这个未参与更新的固定子集上，真实下一字的平均对数概率改善；它不能证明已学会通顺句子。第 50 与 200 步贪心结果都重复逗号，说明当前解码下存在强烈重复偏好。采样文字也尚不通顺，不能只凭出现几个常用字就认定模型已学好语言。应将“验证指标改善”和“可读性达标”分开记录，后续再扩大训练与验证、分析重复问题。

完整参考输出如下：

```text
检查点 step=0 固定验证损失=8.651848
贪心： '春风忏帐朝瑁抅螳赋舌捶樵语捣砌张喜班貔蘅舐缨墦降莩\n擢椁迤膻休腌将隙帘姓容喟楚豂拶骈'
采样： '春风衽殆迎腹匿鱼蛮要哀汶旧隶糠贺永屉甩兼赓箔蚤摄鬃重袒姣浒保噎耻蔺骦拧料涕妆哔涸屁美'
检查点 step=5 固定验证损失=8.629895
贪心： '春风忏狄程瑁菜浙皴滨曩疮衫睁协履墦务燐晶咨嵘破叫琴泣秋蟒炷跋秋泥瑁喔虼炖赫缪迥皇檀喇'
采样： '春风衷殆迈腰匮鲂蜉要哈沏旬隰紧赂毙屁甸册赢箱蛙撩鬒钗裙姻浔信嚏联蔷骸拧既浚妮哩淖屡美'
训练 step=25 更新前批次损失=8.428392 grad_norm=0.498488 scale=1.000000
训练 step=50 更新前批次损失=7.898736 grad_norm=0.545869 scale=1.000000
检查点 step=50 固定验证损失=8.174481
贪心： '春风，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，'
采样： '春风衷歙逛良剑，谒趔呲洼曳黎翩郑沽岌相俸遭绷误旂，餐赘娇溃令噪芽订，授暂滪妾咂潮屉舜'
训练 step=75 更新前批次损失=7.735422 grad_norm=0.631560 scale=1.000000
训练 step=100 更新前批次损失=7.442065 grad_norm=0.599918 scale=1.000000
训练 step=125 更新前批次损失=7.642580 grad_norm=0.403113 scale=1.000000
训练 step=150 更新前批次损失=7.306347 grad_norm=0.447093 scale=1.000000
训练 step=175 更新前批次损失=7.663426 grad_norm=0.493130 scale=1.000000
训练 step=200 更新前批次损失=6.695289 grad_norm=0.599060 scale=1.000000
检查点 step=200 固定验证损失=7.676629
贪心： '春风，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，'
采样： '春风袋朽锯花上，诵跋乞暮惙，筷道是匝爆。邙程诡待，髭趣僭朴。亲胭裸，岘悚来你了板南罽'
四本书实际训练窗口数： {'西游记.txt': 1390, '红楼梦.txt': 996, '水浒全传.txt': 433, '三国演义.txt': 381}
完成更新步数： 200
实验记录路径： D:\projects\playatry\docs\training_runs\lesson29.json
多步训练、固定验证与生成对照记录：完成
```

老师在临时目录完成上述实跑，四个检查点顺序正确，四书实际窗口总数为 3200；重新独立加载 200 步文件，生成与保存记录一致。未在仓库生成这些新权重；最终记录字典仅为排版拆行，语法树与已预跑代码一致。

### 实操与提交

运行 python scripts/learn/13_training_experiment.py。不要同时修改学习率、温度或开头；先保留这一轮对照。请提交四个检查点的验证损失、200 步的贪心/采样文字，以及四本书窗口统计。新增四个权重暂被既有 .gitignore 忽略，待实操复查后会为本课检查点添加明确例外，连同 docs/training_runs/lesson29.json、代码和笔记完整上传 GitHub。后续 CI 的短检查与完整实验运行范围将在复查时明确，避免每次都无意跑很长训练。

回答：一、验证损失从约 8.65 降到 7.68，但贪心仍重复逗号，能否认定已经学会通顺写作？二、为什么比较四个检查点时要固定验证子集、输入开头、温度和采样种子？本课为学生待完成练习，不提前记录其运行成功。

### 第二十九课实操与问答复查：通过

学习者完成完整 200 步并提交四个检查点日志、生成样例和四书窗口统计。老师检查实际脚本、JSON 与权重后，再次执行实际训练脚本；复跑产生的实验记录与学习者提交的 JSON 完全一致。仅清理脚本行尾空白，前后语法树一致，未改训练算法或实验条件。

四个固定验证子集损失为 8.651848157246908、8.629894574483236、8.174481391906738、7.676629384358724。四本书窗口数为西游记 1390、红楼梦 996、水浒全传 433、三国演义 381，合计 3200 个窗口、102400 个预测位置。200 步仍未完成一个 epoch，四书窗口数不均等，不将章节打乱称为等量采样。

第一题学习者回答“否”，正确：当前子集的验证交叉熵下降只支持预测目标有所改善，贪心仍重复逗号、采样仍不通顺，不能认定已经能通顺写作。第二题回答“控制变量”，正确：固定数据与解码条件，才能更清楚比较同一训练轨迹中参数变化的影响。本次未使用测试集，也未扩大为全验证集测评。

四份权重已添加明确的 .gitignore 例外，按每课完整上传约定连同实验 JSON、脚本、模型说明与本笔记纳入仓库；文件大小依次为 2667219、2686086、2687062、2687669 字节，合计 10728036 字节。旧五步权重未覆盖。模型说明记录了各文件对应步数与验证损失。

老师另外增加 scripts/ci/check_lesson29.py，用于仓库产物检查，不是新增学生练习。该检查加载四份已有权重，核对格式、步数、完整词表、1536 个固定验证位置、损失容差与合法生成编号；本地全部通过。没有把受跨平台浮点边界影响的整段生成文字作为跨平台黄金值。CI 运行这个只读检查和既有课程检查，语法检查覆盖新训练脚本，但每次推送不自动重跑完整 200 步实验。

权重 SHA-256：step000 为 630b7ea373a0d5c74544775ae4f34bcd3805ab370a77bd760e7df34c9f5848a5；step005 为 dddc439409f3b78f90df12a0cbb8d5dd24edd4e7db17d171bb75afd154c5d0e1；step050 为 09c1f153c07750ce5230da8ef1fa678108f659eddc1a35e6df2bf3d9335e509e；step200 为 e076bc06fe64ecedac3f7aa5cbb28be6d8f7f315f36ee10455a2a06aa50db033。

第二十九课实操和问答通过。下一阶段可先诊断重复现象，例如建立只利用训练集字符频率的简单基线，再比较固定验证损失，判断当前改进是否已超越基本字频预测；该诊断尚未实施，不直接宣称增加步数就能解决重复。

## 第三十课：字符频率基线与重复逗号诊断

状态：老师已在内存中预检完整新脚本，待学习者新建 scripts/learn/14_frequency_baseline.py 并输入、运行与回答。未代写学习者脚本、未更新 RNN 权重或写入正式诊断报告。模型仍使用上一课已保存的 200 步检查点。

### 什么是字频基线

不看前文，每个位置都使用同一份训练字符概率 p(i)。计数 c_i，目标出现总次数 N，词表大小 V，加一平滑 α=1：p(i)=(c_i+α)/(N+αV)。加的是每个编号的虚拟出现次数，避免未出现编号（包括匹配训练范围中未出现的字符或未知编号）概率为零，使验证损失变成无穷。α 在评估前固定，不用验证数据调优。

若计数 [3,1,0]、V=3，平滑后 [4,2,1]，分母为 7，概率 [4/7,2/7,1/7]。这是闭式计数估计，不需要梯度下降。对真实下一字 y 的损失仍是 -ln p(y)，在相同验证位置取平均，所以可与 RNN 的交叉熵比较。

两条字频基线：一条只统计同一章节顺序前 200 个批次的 y，共 102400 个目标；另一条统计全部训练窗口的 y，共 2450688 个目标。前者与当前 RNN 的训练目标出现范围匹配，后者使用了更多训练目标，必须明确标注，不能据后者胜出就作同等数据量的架构优劣结论。两条都不读取验证目标进行计数。

本课统计窗口目标 y，不是任意原文字符：每章首字符不成为下一字目标、丢弃不足完整窗口的尾部，与 RNN 当前训练任务定义保持一致。重复出现的同一字符每次都计数；词表不因验证数据而改变。

### 第一块：新文件导入与两个函数

```python
from importlib import import_module  # 复用固定词表、窗口、模型和评估接口。
from itertools import islice  # 构造与上一课完全相同的固定验证子集。
import json  # 读取实验配置，并保存本课诊断记录。
import random  # 恢复上一课固定的训练章节顺序。
import numpy as np  # 按编号计数、计算概率和交叉熵。


def smoothed_probabilities(counts, alpha=1.0):  # 给每个编号加虚拟计数，再计算字符频率概率。
    counts = np.asarray(counts, dtype=np.float64)  # 用双精度计算概率，避免整数除法问题。
    assert counts.ndim == 1 and counts.size > 0 and np.isfinite(counts).all() and np.all(counts >= 0), "计数不合法"  # 每个词表编号对应一个非负计数。
    assert np.isfinite(alpha) and alpha > 0, "平滑系数必须是有限正数"  # 正平滑保证没出现的编号也有非零概率。
    return (counts + alpha) / (counts.sum() + alpha * counts.size)  # 所有编号的概率总和为 1。


def frequency_loss(probabilities, batches):  # 不看输入前文，只根据正确目标编号计算字频基线损失。
    assert probabilities.ndim == 1 and np.isfinite(probabilities).all() and np.all(probabilities > 0) and np.isclose(probabilities.sum(), 1.0), "概率分布不合法"  # 对数需要正概率。
    log_probs = np.log(probabilities)  # 同一分布用于每个位置，只需计算一次对数。
    loss_sum = 0.0  # 累计全部预测位置的负对数概率。
    positions = 0  # 累计实际评估位置数。
    for batch in batches:  # 遍历与 RNN 相同的验证批次。
        loss_sum += float((-log_probs[batch["y"]]).sum())  # 每个位置查正确答案对应概率，再取负对数。
        positions += batch["y"].size  # 尾批也按实际位置数统计。
    assert positions > 0, "没有可评估的位置"  # 不对空子集计算平均。
    return loss_sum / positions, positions  # 返回按位置平均的交叉熵及评估规模。
```

np.bincount 会按非负整数编号累计出现次数，minlength=V 保证没出现的编号也保留零计数。frequency_loss 直接按目标编号查 log_probs，不需要为每个验证位置复制一整份 (V,) 概率，也不构造 (B,T,V) 的基线分数数组。

### 第二块：main 中恢复训练顺序，并计数两条范围

```python
def main():  # 建立两条训练集字频基线，诊断已保存的两百步 RNN。
    data = import_module("06_make_windows")  # 复用训练数据、固定词表及窗口函数，首次导入会显示旧检查。
    network = import_module("07_tiny_rnn")  # 只导入模型结构，不重新训练。
    training = import_module("11_train_steps")  # 复用保存权重的加载和固定参数评估。
    experiment = json.loads((data.ROOT / "docs" / "training_runs" / "lesson29.json").read_text(encoding="utf-8"))  # 读取上一课实际实验条件。
    config = experiment["config"]  # 保持训练顺序、批大小和窗口长度与上一课一致。
    assert experiment["train_sha256"] == data.vocab["train_sha256"], "训练数据版本不一致"  # 不能将不同数据版本的基线直接比较。
    records = list(data.train_records)  # 复制训练章节列表，原数据顺序保持不变。
    random.Random(config["chapter_shuffle_seed"]).shuffle(records)  # 恢复上一课训练时的同一章节顺序。
    vocab_size = len(data.tokens)  # 每个词表编号都要有一个计数和概率。
    full_counts = np.zeros(vocab_size, dtype=np.int64)  # 统计全部训练窗口目标字符。
    seen_counts = np.zeros_like(full_counts)  # 另统计 RNN 前两百步实际见过的目标字符。
    batches = data.iter_batches(data.iter_windows(records, config["seq_length"], config["seq_length"]), config["batch_size"])  # 重走训练窗口流，只计数不更新模型。
    for batch_index, batch in enumerate(batches, start=1):  # 按相同顺序遍历全部训练批次。
        batch_counts = np.bincount(batch["y"].reshape(-1), minlength=vocab_size)  # 同一编号出现多次就增加多次计数。
        full_counts += batch_counts  # 全训练集基线累加每个批次的目标出现次数。
        if batch_index <= config["steps"]:  # 只给匹配训练范围的基线累计前两百批。
            seen_counts += batch_counts  # 与 RNN 相同的目标出现范围，包含重复字符。
    assert int(seen_counts.sum()) == 102400 and int(full_counts.sum()) == 2450688, "本课目标计数与实验范围不符"  # 确认匹配范围和全训练范围。
    alpha = 1.0  # 预先固定加一平滑，不用验证集选择该数值。
    seen_probs = smoothed_probabilities(seen_counts, alpha)  # 得到匹配两百步目标范围的频率模型。
    full_probs = smoothed_probabilities(full_counts, alpha)  # 得到使用全部训练目标的频率模型。
```

此处遍历全部批次只为了计数，不训练模型。前 200 批的匹配顺序从上一课 JSON 配置恢复，不能随意用未打乱的前 200 批替代。词频拟合使用 train_records；验证集到下一块才被读取用于损失测量。

### 第三块：评估相同验证子集，并观察高频字符

```python
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 验证集仅用于测量，不参与计数拟合。
        validation_records = [json.loads(line) for line in file]  # 保留原验证章节顺序。
    validation_batches = list(islice(data.iter_batches(data.iter_windows(validation_records, config["seq_length"], config["seq_length"]), config["batch_size"]), config["validation_batches"]))  # 使用与上一课相同的前三批。
    seen_loss, seen_positions = frequency_loss(seen_probs, validation_batches)  # 测量匹配训练范围的字频模型。
    full_loss, full_positions = frequency_loss(full_probs, validation_batches)  # 测量看过更多训练目标的全训练集字频模型。
    reference = next(row for row in experiment["history"] if row["step"] == config["steps"])  # 找到上一课两百步检查点记录。
    model, vocab, step = training.load_checkpoint(data.ROOT / "model" / reference["checkpoint"], network)  # 加载已训练模型，不增加训练步数。
    assert vocab == data.vocab and step == config["steps"], "权重与实验配置不一致"  # 词表顺序和训练步数都要匹配。
    rnn_loss, rnn_positions = training.evaluate(model, validation_batches, network)  # 固定 RNN 参数，重新评估同一子集。
    assert seen_positions == full_positions == rnn_positions == reference["validation_positions"] == 1536, "评估范围不一致"  # 四种预测方式必须比较相同位置。
    assert np.isclose(rnn_loss, reference["validation_loss"], rtol=1e-6, atol=1e-6), "RNN 与历史记录不一致"  # 先确认加载结果可复核。
    metrics = {"uniform": float(np.log(vocab_size)), "matched_frequency": seen_loss, "full_frequency": full_loss, "rnn200": rnn_loss}  # 均匀预测的交叉熵为 ln(V)，其余使用同一验证子集。
    print("匹配前两百步的目标字符数：", int(seen_counts.sum()))  # 应为 102400。
    print("全部训练窗口的目标字符数：", int(full_counts.sum()))  # 应为 2450688，区别于原文字符数。
    for name, loss in metrics.items():  # 列出四种预测方式的同范围损失。
        print(f"{name} 固定验证交叉熵：{loss:.6f}")  # 交叉熵越低，在当前目标上的平均对数概率越好。
    top_frequency = []  # 保留全训练集最常见的五个编号，供解释与记录。
    for index in np.argsort(full_probs)[-5:][::-1]:  # 按概率从高到低取前五项。
        item = {"token": data.tokens[index], "count": int(full_counts[index]), "probability": float(full_probs[index])}  # 记录真实计数和加一平滑概率。
        top_frequency.append(item)  # 将观察保存在本课报告中。
        print("高频字符：", repr(item["token"]), "次数：", item["count"], "平滑概率：", round(item["probability"], 6))  # 用 repr 显示标点和可能的换行。
    most_frequent = data.tokens[int(np.argmax(full_probs))]  # 字频模型不看前文，所以每步的最高概率编号都相同。
    print("全训练集字频基线贪心：", repr("春风" + most_frequent * 40))  # 演示简单字频模型也会连续重复最高频字符。
```

均匀预测在 V 个编号上都给 1/V，任意正确目标的损失为 ln(V)，所以可以直接计算 uniform 指标。所有预测方式都针对同一 1536 个验证位置，不能把训练集频率拟合损失与 RNN 验证损失混比。

字频模型不看前文，最高频字符在每一步都相同；当前最高频目标是中文逗号。其贪心生成自然重复逗号，说明重复现象可以在一个实现正确但不利用上下文的简单模型中出现；不能仅凭重复就认定 RNN 保存加载或反向传播有错。

### 第四块：查看 RNN 对不同输入的真实分布

```python
    observations = []  # 记录不同输入下 RNN 的实际下一字分布摘要。
    comma_id = data.char_to_id["，"]  # 找到当前词表中中文逗号的编号。
    for text in ("春风", "秋风", "唐僧", "曹操", "宝玉", "春风，", "春风，，"):  # 固定若干输入，包含相同末字与逐步追加逗号的对照。
        x = np.array([[data.char_to_id.get(char, data.unk_id) for char in text]], dtype=np.int64)  # 按同一词表编码每条输入。
        probabilities = network.softmax(model.forward(x)[0, -1, :])  # 只前向，查看读完输入后的下一字概率。
        next_id = int(np.argmax(probabilities))  # 观察贪心选择，不改变模型参数。
        item = {"input": text, "top_token": data.tokens[next_id], "top_probability": float(probabilities[next_id]), "comma_probability": float(probabilities[comma_id])}  # 保存最高候选和逗号概率。
        observations.append(item)  # 将不同输入下的观察保留在报告中。
        print(f"输入={text!r} 最高候选={item['top_token']!r} 最高概率={item['top_probability']:.6f} 逗号概率={item['comma_probability']:.6f}")  # 不能把最高候选误认为概率接近百分之百。
```

比较“春风”和“秋风”时末字相同，较早输入不同；概率变化可说明输出受到输入历史变化影响，但不证明已经学会语法或正确语义。依次给“春风”“春风，”“春风，，”时，观察最高候选及逗号概率，联系生成循环中把自己选择的字反馈进输入这一机制。只是固定少量输入的观察，不将其推广为所有前文的完整因果分析。

### 第五块：保存诊断记录并添加入口

```python
    result = {  # 汇总本课可复查的诊断结果。
        "alpha": alpha, "train_sha256": data.vocab["train_sha256"],  # 记录平滑设置与训练来源。
        "matched_steps": config["steps"], "matched_target_count": int(seen_counts.sum()), "full_target_count": int(full_counts.sum()),  # 标明两条基线不同的计数范围。
        "validation_positions": rnn_positions, "reference_checkpoint": reference["checkpoint"],  # 记录共同评估范围和被诊断的 RNN。
        "metrics": metrics, "top_frequency": top_frequency, "context_observations": observations,  # 保留损失、字频和真实输入对照。
    }  # 本课只写诊断记录，不保存或更新新的 RNN 权重。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson30.json"  # 与上一课实验记录分开保存。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保记录目录存在。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的诊断结果。
    print("字频基线诊断记录：", report_path)  # 显示本课记录文件位置。
    print("训练集计数、同范围验证与字频基线检查：通过")  # 表示比较流程正确，不强制 RNN 必须胜出。


if __name__ == "__main__":  # 直接运行文件时才开始基线诊断。
    main()  # 只统计训练目标并评估，不执行参数更新。
```

### 老师预检结果与解释

| 预测方式 | 同一固定验证子集交叉熵 |
|---|---:|
| uniform | 8.621013 |
| matched_frequency | 6.890807 |
| full_frequency | 6.662431 |
| rnn200 | 7.676629 |

这里同范围字频约 6.89081，低于 RNN 的 7.67663，说明当前 200 步 RNN 在这个固定子集上尚未超越简单字频预测。该结论不等于 RNN 架构无法学习上下文，也不等于当前 RNN 完全没使用输入；它只说明目前训练状态仍不足。全训练集字频约 6.66243，还利用了更多训练目标，需要单独看待。

同一 RNN 已比均匀预测好，但仍比同范围字频差，这比单独看到损失随步数下降提供了更具体的参照。最高候选是逗号也不代表概率接近 100%：当前“春风”后的逗号概率约 0.00694，即约 0.694%；输入追加逗号后，它在这些示例中的概率会上升。贪心只看谁最大，即使最高概率很小，也会选择它；反馈后的输入仍让它居首，就会继续重复。

计数与基线概率存储主要为 O(V)，遍历目标计数主要为 O(N)，每个 NumPy 批次还需构造 V 长度的 bincount 临时数组。基线评估约 O(M)，M 为验证目标数，无需循环隐藏状态。该实现的全训练流计数会多次分配 V 长度临时数组，优先保证清晰，不声称已经最优。模型参数没有增加，本课没有反向传播。

老师在临时目录运行上述完整代码，所有计数、相同评估位置与 RNN 历史指标核对通过；另检查 [3,1,0] 的精确加一平滑、零计数均匀分布、大小不同批次按位置平均。未生成正式报告或改写权重。参考输出如下：

```text
匹配前两百步的目标字符数： 102400
全部训练窗口的目标字符数： 2450688
uniform 固定验证交叉熵：8.621013
matched_frequency 固定验证交叉熵：6.890807
full_frequency 固定验证交叉熵：6.662431
rnn200 固定验证交叉熵：7.676629
高频字符： '，' 次数： 180616 平滑概率： 0.073534
高频字符： '。' 次数： 81208 平滑概率： 0.033062
高频字符： '：' 次数： 37689 平滑概率： 0.015345
高频字符： '“' 次数： 34183 平滑概率： 0.013917
高频字符： '”' 次数： 34162 平滑概率： 0.013909
全训练集字频基线贪心： '春风，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，'
输入='春风' 最高候选='，' 最高概率=0.006939 逗号概率=0.006939
输入='秋风' 最高候选='，' 最高概率=0.005782 逗号概率=0.005782
输入='唐僧' 最高候选='，' 最高概率=0.006223 逗号概率=0.006223
输入='曹操' 最高候选='，' 最高概率=0.007650 逗号概率=0.007650
输入='宝玉' 最高候选='，' 最高概率=0.007245 逗号概率=0.007245
输入='春风，' 最高候选='，' 最高概率=0.034841 逗号概率=0.034841
输入='春风，，' 最高候选='，' 最高概率=0.057523 逗号概率=0.057523
字频基线诊断记录： D:\projects\playatry\docs\training_runs\lesson30.json
训练集计数、同范围验证与字频基线检查：通过
```

### 实操与提交

运行 python scripts/learn/14_frequency_baseline.py，贴出四种损失、最高频字符和“春风”“春风，”“春风，，”的观察。报告会写入 docs/training_runs/lesson30.json，不覆盖 lesson29.json 或权重。完成后检查实际脚本与报告、补充笔记、加入 CI 并同步 GitHub。

回答：一、为什么字频计数只能使用训练集，不能把验证集目标也计进去？二、当前 RNN 损失约 7.68，同范围字频约 6.89，哪个在这个固定子集上的交叉熵更好？未完成前不记录为学生已掌握。

### 第三十课实操复查：通过，拟合与验证标签用途继续澄清

老师检查学习者实际脚本并重新运行，训练目标计数、共同评估范围和历史 RNN 指标核对全部通过。匹配范围与全训练目标数分别为 102400、2450688，共同验证位置数为 1536；四种交叉熵依次为 uniform 8.621012520055235、matched_frequency 6.890807442455625、full_frequency 6.662431022714794、rnn200 7.676629384358724，与预检一致。最高频目标为中文逗号，计数 180616，平滑概率约 0.073534。

实际输入“春风”“秋风”的下一字逗号概率分别约 0.006939、0.005782；“春风，”“春风，，”分别约 0.034841、0.057523。记录支持当前输入下的概率和贪心反馈分析，不扩展为所有上下文的语义结论。没有重新训练或修改权重。仅清理脚本行尾与末尾空白，语法树保持一致，未改变计算逻辑。脚本与 docs/training_runs/lesson30.json 纳入仓库，CI 增加本课计数与评估流程，不运行新训练。

第二题回答“匹配范围字频”，正确：在相同验证子集上 6.890807 小于 7.676629，因此这条字频基线的交叉熵更好。该结论限定为当前模型、当前子集和当前训练状态，不代表 RNN 架构永远不如字频。

第一题回答“现在是检查训练结果”，说到了验证的用途，但仍需明确禁止验证计数的原因。字符计数本身就是拟合：计数 c_i 决定概率 p_i=(c_i+α)/(N+αV)，即使没有反向传播，加入新的答案也会改变学到的预测分布。如果把验证集目标加入计数，再用同一验证集测量，验证答案就参与了拟合，结果不再是对未参与拟合数据的独立评估。不能只将“训练”理解为使用梯度下降。

验证标签可以用于计算 -ln p(y) 和评估平均损失，这是测量；不能用于增加 c_i、重估概率，这是拟合。当前实际代码正确地区分了两者：训练 batch["y"] 用于 bincount，验证 batch["y"] 只用于查已固定概率计算损失。

本课实操和第二题通过，第一题补充解释后待复核。下一次只确认：如果把验证集中的逗号次数也加进 full_counts，再用这个验证集测量损失，是否仍属于独立验证？本次不推进下一课。

### 独立验证复核纠错：验证答案不能加入拟合计数

学习者回答“算”，此处应为“不算”。只要把验证答案计入 full_counts，它们就参与决定概率分布，之后在同一验证数据上计算损失，不再是对未参与拟合数据的独立测量。是否进行了反向传播、是否修改 RNN 权重，都不是唯一判断标准；本课字频模型的计数本身就是拟合信息。

简化例子：训练计数为“甲”3次、“乙”1次，固定加一平滑后，P(乙)=(1+1)/(4+2)=1/3。假设验证集有三个目标都是“乙”，若偷偷加入计数，P(乙)就变成 (4+1)/(7+2)=5/9；随后对这三个“乙”评估，概率已经利用了它们自己的答案。这说明模型预测分布因验证数据而变，不能仍称独立验证。例子只用于说明数据用途，不是本项目实际字符计数。

正确顺序是用训练计数固定概率，再用验证答案只计算 -ln p(y)，评估过程中不改变计数或概率。能读取验证答案计算损失，不等于可以用它更新拟合结果。本次只补记纠错，未修改代码或权重，也不推进下一课。

下一次只核对直接影响：把验证集里的“乙”加入计数后，P(乙)从 1/3 变成 5/9，这份预测概率是否受到了验证答案的影响？等待学习者回答，暂不标记概念已掌握。

### 验证答案影响概率复核：回答正确

学习者回答“有”，正确：把验证目标加入计数后，拟合出的概率就受到了验证答案的影响，再用同一批目标测量不属于独立验证。字频模型应只用训练集计数固定概率，验证标签仅用于计算损失，不改变该概率分布。

第三十课代码、基线比较与实操检查通过；学习者已正确选出匹配范围字频基线的较低交叉熵，并在补充例子后确认验证答案会影响拟合概率。独立验证的区别后续实践中继续巩固，不据单项复核声称已掌握全部数据泄漏情形。本次仅补记回答，没有修改代码、权重或重新运行实验。

## 第三十一课：单窗口拟合检查（小样本过拟合测试）

状态：老师已在临时目录预检以下完整脚本，待学习者新建 scripts/learn/15_overfit_one_window.py 并输入、运行与回答。未代写学习者脚本，没有覆盖旧 RNN 检查点。本课是独立诊断，不是新的全训练集模型对照。

### 为什么先检查能否记住一个窗口

上一课发现 200 步 RNN 尚未超过同训练目标范围的字频基线。可能原因包括训练不充分、更新设置等，不能仅凭这一结果就判定架构或反向实现错误。本课固定一个真实训练窗口，反复优化同一个目标，检查模型与数据、损失、梯度和更新是否能够配合拟合极小样本。

这里“过拟合测试”指主动尝试把很小的训练样本记住，是排查训练流程的诊断手段，不是以记住样本作为泛化目标。通过只支持在这个样本与配置上能够有效拟合，不能证明所有实现边界都正确；若失败也要进一步检查数据、学习率、模型容量与数值行为，不能自动认定某一处代码必错。

取训练集首个窗口，B=1、T=32，重复更新 300 次；模型从相同种子新建，不从旧两百步权重继续训练。每步梯度的平均仍除以 B*T=32，不再除以重复次数。实际只有一个独立窗口、32 个不同的预测位置，被重复使用 300 次，合计 9600 次位置使用，不等于 300 个不同窗口。模型参数依旧为 719723。

### 第一块：新文件导入与测量函数

```python
from importlib import import_module  # 复用数据、模型、梯度裁剪、保存和生成函数。
import json  # 保存单窗口拟合的配置、指标与生成结果。
import numpy as np  # 检查标签、计算逐位置预测正确数。

network = import_module("07_tiny_rnn")  # 导入完整模型与裁剪工具，不执行旧演示。
training = import_module("11_train_steps")  # 复用权重保存与加载，不执行旧训练入口。
decoding = import_module("12_generate_text")  # 复用逐字生成，不执行旧生成演示。


def measure_fit(model, x, y, step):  # 在同一个训练窗口上测量当前拟合程度。
    logits = model.forward(x)  # 用当前参数计算该窗口各位置的下一字分数。
    loss, _ = network.cross_entropy(logits, y)  # 计算这个训练窗口的平均交叉熵，不是验证损失。
    correct = int((np.argmax(logits, axis=-1) == y).sum())  # 每个位置取最高分编号，与真实下一字比较。
    print(f"step={step} 固定训练窗口损失={loss:.6f} 正确位置={correct}/{y.size}")  # 日志比较的是同一组输入和目标。
    return {"step": step, "loss": loss, "correct": correct, "positions": int(y.size)}  # 记录拟合过程，供之后复查。
```

correct 是在真实输入 x 的各个位置取最高分编号，与真实目标 y 比较的训练正确数。它不是自由生成的正确率，也不是验证集指标。measure_fit 固定参数只测量当前训练窗口，随后训练循环继续使用同一 x、y。

### 第二块：main 中固定唯一窗口并初始化

```python
def main():  # 用一个真实窗口进行小样本拟合诊断。
    data = import_module("06_make_windows")  # 复用训练集首窗口和固定词表，首次导入仍会显示旧数据检查。
    seq_length = 32  # 只学习一个包含 32 个预测位置的窗口。
    max_steps = 300  # 对同一窗口反复更新三百次。
    learning_rate = 0.1  # 使用当前普通梯度下降学习率。
    max_norm = 1.0  # 保留全局梯度裁剪。
    batch = next(data.iter_batches(data.iter_windows(data.train_records, seq_length, seq_length), 1))  # 只取一个窗口，此后不再获取新批次。
    x, y = batch["x"], batch["y"]  # 固定输入与向后错位一字的目标。
    assert x.shape == y.shape == (1, seq_length) and np.array_equal(x[:, 1:], y[:, :-1]), "诊断窗口格式错误"  # 确认任务仍是下一字预测。
    saved_x, saved_y = x.copy(), y.copy()  # 保存数据副本，检查三百步使用的是同一份样本。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 新建独立诊断模型，不修改上一课的两百步模型。
    history = [measure_fit(model, x, y, 0)]  # 训练前记录同一窗口的基线。
```

next 只调用一次，循环里不再取新 batch。保存 x、y 副本只是检查数据没有被意外修改。当前窗口来源为 三国演义:010、起点 0，这个样本是训练集内容，不读取验证或测试数据用于更新。

### 第三块：接在 main 内反复训练同一个窗口

```python
    for step in range(1, max_steps + 1):  # 每步仍然使用相同的 x 和 y。
        loss, raw_grads = model.loss_and_grad(x, y)  # 在固定训练窗口上求完整梯度。
        assert np.isfinite(loss), "训练损失异常"  # 无效数值不能继续用于更新。
        grads, _, _ = network.clip_gradients(raw_grads, max_norm)  # 对全部共享参数梯度做一次全局裁剪。
        for name in model.params:  # 完整反向结束后统一更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 新参数继续用于下一次相同样本的学习。
        assert all(np.isfinite(param).all() for param in model.params.values()), "参数更新异常"  # 确认拟合过程数值有限。
        if step in (20, 100, 300):  # 在预先指定的步数测量同窗口更新后的结果。
            history.append(measure_fit(model, x, y, step))  # 既记录损失，也记录正确位置数。
    assert np.array_equal(x, saved_x) and np.array_equal(y, saved_y), "训练样本被意外改变"  # 防止误以为固定样本，实际却改了数据。
    assert history[-1]["loss"] < history[0]["loss"] * 0.1 and history[-1]["correct"] == y.size, "本例单窗口拟合未达到预期"  # 对当前配置检查损失显著下降且三十二个位置全对。
```

固定目标为 L(θ)=-(1/32)Σ_t ln p_θ(y_t|x_≤t)，每一步在新的参数点计算同一数据目标的梯度。保持模型在循环外创建，完整反向后裁剪，再更新五组参数。指定步数的测量发生在该步更新后，前后都是同一组位置，能观察拟合进度，但不要求每一步损失必然单调下降。

这里的显著下降、32/32 以及后面的生成完全匹配，是对本课固定数据与配置的具体诊断预期，不是声称任意窗口或任意超参数都能在 300 步达到相同水平。

### 第四块：只给首字，检查能否复现训练片段

```python
    prompt = data.tokens[int(x[0, 0])]  # 只给这个训练窗口的第一个字符作为生成开头。
    expected_text = prompt + "".join(data.tokens[int(index)] for index in y[0])  # 拼出该窗口对应的完整三十三字训练片段。
    generated_text, _ = decoding.generate(model, data.vocab, prompt, new_tokens=seq_length, context_size=seq_length)  # 不提供后续正确字符，让模型贪心生成三十二步。
    assert generated_text == expected_text, "本例尚不能从首字复现训练片段"  # 这只检查已训练片段的记忆，不证明泛化能力。
    checkpoint_path = data.ROOT / "model" / "lesson31_single_window.npz"  # 用独立名称保存诊断权重，不覆盖全语料实验检查点。
    training.save_checkpoint(checkpoint_path, model, data.vocab, max_steps)  # 保存只反复学过一个窗口的模型。
    print("训练片段：", repr(expected_text))  # 保留换行与空格，便于逐字核对。
    print("只给首字后的生成：", repr(generated_text))  # 当前例子应与训练片段相同。
```

prompt 是“第”；expected_text=首字符+32 个真实下一字目标，共 33 个字符。生成只接收首字符，不把后续目标喂给生成函数，输出仍需一步一步把自身预测加入上下文。本次生成 32 步、上下文上限 32，整个片段在当前窗口定义内。生成成功只证明可以复现已经反复见过的训练片段，不证明能写未见故事或掌握长距离语法。

### 第五块：保存独立诊断记录与入口

```python
    result = {  # 保存诊断条件与观察，不把训练指标写成验证指标。
        "config": {"steps": max_steps, "seq_length": seq_length, "batch_size": 1, "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42},  # 记录固定训练设置。
        "train_sha256": data.vocab["train_sha256"], "chapter_id": batch["chapter_ids"][0], "start": batch["starts"][0],  # 记录唯一训练窗口的来源。
        "input_ids": x.tolist(), "target_ids": y.tolist(), "history": history,  # 保留输入、真实目标和同窗口拟合指标。
        "prompt": prompt, "expected_training_text": expected_text, "generated_training_text": generated_text, "checkpoint": checkpoint_path.name,  # 标明生成结果对应的是已训练片段。
    }  # 本次没有使用验证集或测试集进行拟合。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson31.json"  # 将本课诊断记录与之前实验分开。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 创建记录目录，已有目录继续使用。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的实验记录。
    print("诊断权重：", checkpoint_path)  # 文件只用于单窗口记忆检查，不能作为泛化已达标的证据。
    print("诊断记录：", report_path)  # 显示可复查的配置与指标文件。
    print("单窗口拟合、数据固定与训练片段复现检查：通过")  # 说明当前小样本诊断通过。


if __name__ == "__main__":  # 直接运行脚本时才执行单窗口实验。
    main()  # 从头训练独立的小样本诊断模型。
```

权重文件名明确标为单窗口，避免误当成新全语料模型替换 lesson29_step200.npz。元数据中仍有完整固定词表，但本次更新只用了这个窗口。脚本重复运行仍是从新模型开始重复该实验，不是继续训练旧文件。

### 老师预检观察与解释

| 已完成更新步数 | 固定训练窗口损失 | 正确位置 |
|---:|---:|---:|
| 0 | 8.747080 | 0/32 |
| 20 | 7.410583 | 7/32 |
| 100 | 1.259116 | 32/32 |
| 300 | 0.071682 | 32/32 |

只给首字后能复现：

```text
'第十回\u3000勤王室马腾举义\u3000报父仇曹操兴师\n却说李、郭二贼欲杀献帝，张'
```

当前模型已经能记住这一小段。虽然这一结果支持当前小样本训练链路可以工作，但不能据此认为此前全语料训练已经充分或泛化达标。本课约 0.07168 是训练窗口损失，不能与上一课 1536 个验证位置上的约 7.67663 或基线约 6.89081 直接比较好坏；数据与模型训练任务范围都不同。

每步 B 从之前的 16 变成 1，单步工作量和输出数组显著减少。主要计算约 O(300*T*(H²+HV))，单份 logits 为 (1,32,5547)，float32 数据区 710016 字节。没有增加模型参数；只保存最后一个诊断权重与小 JSON。完整窗口的充分训练仍是后续任务，这个检查不是通顺文本生成的最终目标。

老师在临时目录运行全部代码，固定数据与拟合断言通过；另独立加载保存模型，重算 loss 和 32/32 正确数并再次生成，结果与训练记录一致，没有改写仓库中的旧权重。参考输出：

```text
step=0 固定训练窗口损失=8.747080 正确位置=0/32
step=20 固定训练窗口损失=7.410583 正确位置=7/32
step=100 固定训练窗口损失=1.259116 正确位置=32/32
step=300 固定训练窗口损失=0.071682 正确位置=32/32
训练片段： '第十回\u3000勤王室马腾举义\u3000报父仇曹操兴师\n却说李、郭二贼欲杀献帝，张'
只给首字后的生成： '第十回\u3000勤王室马腾举义\u3000报父仇曹操兴师\n却说李、郭二贼欲杀献帝，张'
诊断权重： D:\projects\playatry\model\lesson31_single_window.npz
诊断记录： D:\projects\playatry\docs\training_runs\lesson31.json
单窗口拟合、数据固定与训练片段复现检查：通过
```

### 实操与提交

运行 python scripts/learn/15_overfit_one_window.py，提交四个步数的训练窗口损失、正确位置数和最终生成结果。代码将写入 model/lesson31_single_window.npz 与 docs/training_runs/lesson31.json。当前新权重默认被忽略；实操复查后将为本课诊断文件添加明确例外，连同代码、报告和笔记一起上传。CI 范围会在复查时记录清楚。

回答：一、同一个窗口重复更新 300 次，独立训练窗口是 1 个还是 300 个？二、从首字复现这段训练文字，是否证明模型已经会写没有见过的新故事？本课实操待学习者完成，不提前记为已掌握。

### 第三十一课实操与问答复查：通过

学习者完成诊断并提交四行指标、生成文本与文件路径。老师检查实际脚本与报告，重新完整执行 300 步单窗口实验，生成的 JSON 与学习者提交的记录完全一致，未修改学习者代码。四个阶段的损失依次为 8.747079849243164、7.410582542419434、1.2591156959533691、0.07168249785900116，正确位置数为 0、7、32、32，目标总位置数始终为 32。

只给首字“第”，可以复现训练窗口的完整 33 字片段。另在新的 Python 进程中独立加载保存文件，再次计算训练窗口损失、32/32 正确数并生成，结果与记录一致。实际保存路径为 model/lesson31_single_window.npz，没有多余目录或分隔符；用户消息中展示的路径不影响实际文件。

第一题回答“独立训练窗口是1个”，正确：三百次更新重复使用同一 x、y，没有增加不同训练样本。第二题回答“不能证明”，正确：复现已反复训练的片段只支持记忆与拟合，不证明会写未见故事或泛化表现达标。本课约 0.07168 是训练窗口损失，仍不能与上一课验证子集的 7.67663 或字频基线 6.89081 直接比较。

本课代码、JSON 与诊断权重按完整上传约定纳入仓库；为该权重添加明确 .gitignore 例外并更新模型说明。文件大小 2686819 字节，SHA-256 为 4d0732b189ff55d3b349a1f07ee41543cba1211c1eb4052014ca7fce6c4b6c6f。旧检查点和训练数据未修改。

CI 增加实际运行该单窗口 300 步诊断及复现检查，单步批大小为 1；第二十九课仍只读校验四份已保存检查点，不自动重跑两百步多批实验。第三十一课实操与问答均通过。下一阶段可回到四书语料，规划完整一轮训练与固定验证、日志、检查点记录，再与字频基线比较；这项完整训练尚未实施。

## 第三十二课：四书完整一轮训练、尾批与同范围字频比较

状态：老师已在临时目录完成整轮预跑及完整验证，待学习者新建 scripts/learn/16_train_one_epoch.py 并输入运行。未替学习者创建脚本，未在仓库保存本课新权重或报告。旧模型、单窗口诊断和之前课程文件保持不变。

### 本课范围与控制条件

从种子 42 的新模型开始，章节按种子 42 打乱一次，B=16、T=32、SGD 学习率 0.1、全局裁剪阈值 1，沿用第 29 课前两百步的训练顺序与结构。模型只创建一次，训练生成器不再用 islice 截断，直到自然耗尽。只有一个 epoch，不加载 lesson31_single_window.npz，也不把单样本记忆模型当作全语料起点。

整轮窗口数 76584，更新步数 ceil(76584/16)=4787，最后一批 8 个窗口；共 2450688 个训练预测位置。尾批目标为 8*32=256 个，loss_and_grad 自动按实际 y.size 归一化，不固定除以 512。

中途固定在 0、200、1000、3000 和最后一步评估原验证前三批（1536 个位置）并生成同样设置的文字；末尾用最终参数另外评估完整验证窗口集。完整验证窗口为 9905 个、620 批，最后一批 1 个窗口，目标位置总数 316960。原验证文本 317717 个字符，但按章首字和短尾规则，并非每个原文字都成为目标。

### 第一块：新文件导入

```python
from importlib import import_module  # 复用已有数据、训练、生成和字频基线代码。
from itertools import islice  # 只缓存中途评估用的固定验证前三批。
from time import perf_counter  # 测量训练与周期检查点所需的实际时间。
import json  # 保存本轮配置、覆盖统计和评估记录。
import random  # 用固定种子打乱训练章节顺序。
import numpy as np  # 累加目标计数，并检查训练数值。

network = import_module("07_tiny_rnn")  # 复用完整 RNN 和全局梯度裁剪。
training = import_module("11_train_steps")  # 复用固定参数评估与权重保存。
decoding = import_module("12_generate_text")  # 复用相同开头下的贪心和采样生成。
baseline = import_module("14_frequency_baseline")  # 复用加一平滑和字频损失，不执行旧诊断入口。
```

旧模块有 main 保护，导入不会重新执行它们的训练入口。第六课数据模块仍会先运行已有顶层数据检查。本课调用字频函数只是计算概率和损失，全部计数来自训练循环里的 y。

### 第二块：添加检查点记录函数

```python
def record_checkpoint(model, vocab, batches, step, output_dir):  # 测量固定子集、生成样例并保存本步权重。
    loss, positions = training.evaluate(model, batches, network)  # 参数固定，只评估此前同一份验证子集。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 固定开头与贪心设置。
    sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定采样温度和种子。
    path = output_dir / f"lesson32_step{step:04d}.npz"  # 用本课独立名称保存，避免覆盖旧实验。
    training.save_checkpoint(path, model, vocab, step)  # 保存完整五组参数和配套词表。
    print(f"检查点 step={step} 固定验证子集损失={loss:.6f}", flush=True)  # 长训练时立即显示检查点进度。
    print("贪心：", repr(greedy), flush=True)  # 原样记录生成质量，包括重复。
    print("采样：", repr(sampled), flush=True)  # 不只挑选看起来好的样例。
    return {"step": step, "validation_loss": loss, "validation_positions": positions, "greedy": greedy, "sampled": sampled, "checkpoint": path.name}  # 返回本次检查点记录。
```

五份文件名为 lesson32_step0000.npz、lesson32_step0200.npz、lesson32_step1000.npz、lesson32_step3000.npz、lesson32_step4787.npz。路径与之前实验不同，不覆盖旧模型。它们记录同一次训练轨迹上的参数，不是五次从头重训。没有验证效果必须逐次更好的断言，也不宣称最后一步必然是最优模型。

### 第三块：main 中准备整轮窗口与固定验证子集

```python
def main():  # 从初始模型开始，完整遍历一次四书训练窗口集。
    data = import_module("06_make_windows")  # 复用训练章节、词表与按章窗口，首次导入仍显示旧检查。
    batch_size, seq_length = 16, 32  # 保持每批最多十六个窗口、每窗口三十二个预测位置。
    learning_rate, max_norm = 0.1, 1.0  # 与两百步实验保持相同更新设置。
    records = list(data.train_records)  # 复制章节列表，不改动磁盘数据。
    random.Random(42).shuffle(records)  # 沿用相同的固定章节打乱顺序。
    expected_windows = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in records)  # 按相同窗口规则计算整轮样本数。
    expected_steps = (expected_windows + batch_size - 1) // batch_size  # 向上取整，最后的小批次也更新一次。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取独立验证章节，不用于参数更新。
        validation_records = [json.loads(line) for line in file]  # 保留验证章节顺序。
    subset = list(islice(data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size), 3))  # 中途仍只测同样的前三批。
    assert len(subset) == 3, "固定验证子集不足三批"  # 不允许悄悄缩小原有比较范围。
    model = network.TinyRNN(len(data.tokens), hidden_size=64, seed=42)  # 新建一次模型，从头训练完整一轮，不加载单窗口诊断权重。
    output_dir = data.ROOT / "model"  # 本课检查点使用独立文件名保存。
    history = [record_checkpoint(model, data.vocab, subset, 0, output_dir)]  # 记录尚未更新的零步基线。
    checkpoints = {200, 1000, 3000, expected_steps}  # 按预先固定的步数记录参数快照。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 用章节标识追踪四书训练覆盖。
    book_windows = {record["book"]: 0 for record in records}  # 为每本书累计实际训练窗口数。
    full_counts = np.zeros(len(data.tokens), dtype=np.int64)  # 只用本轮训练目标拟合全训练集字频基线。
    seen_windows, seen_positions, completed_steps, last_batch_size = 0, 0, 0, 0  # 初始化整轮覆盖统计。
    loss_sum = 0.0  # 累计各批更新前的损失总和。
    batches = data.iter_batches(data.iter_windows(records, seq_length, seq_length), batch_size)  # 这次不再用 islice 截断训练批次流。
    print("计划整轮窗口数与更新步数：", expected_windows, expected_steps, flush=True)  # 当前应为 76584 和 4787。
    started = perf_counter()  # 开始计时，后续耗时包含周期检查点的评估、生成和保存。
```

expected_windows 使用已有章节内 L+1 编号构造规则计算；expected_steps 用向上取整保留尾批。中途只缓存三个验证批次，完整训练窗口仍按生成器逐批提供。

### 第四块：main 内完整训练循环与覆盖核对

```python
    for step, batch in enumerate(batches, start=1):  # 直到训练窗口生成器自然结束，完整处理一次。
        x, y = batch["x"], batch["y"]  # 读取本批实际行数，最后一批只有八行。
        loss, raw_grads = model.loss_and_grad(x, y)  # 梯度按本批实际位置数平均，不假定始终有 512 个位置。
        assert np.isfinite(loss), "训练损失异常"  # 出现无效数值时停止。
        grads, raw_norm, scale = network.clip_gradients(raw_grads, max_norm)  # 完整反向后执行一次全局裁剪。
        for name in model.params:  # 使用当前批次的完整梯度更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 下一批沿用更新后的参数。
        assert all(np.isfinite(param).all() for param in model.params.values()), "更新后的参数异常"  # 防止无效权重继续传播。
        full_counts += np.bincount(y.reshape(-1), minlength=len(data.tokens))  # 同时累计全轮训练目标的真实字频。
        seen_windows += x.shape[0]  # 尾批按实际八行统计，不补成十六行。
        seen_positions += y.size  # 累计实际预测位置数。
        loss_sum += loss * y.size  # 按位置数加权，避免尾批影响平均方式。
        completed_steps, last_batch_size = step, x.shape[0]  # 记录最新步数与最后批次大小。
        for chapter_id in batch["chapter_ids"]:  # 将每行训练窗口归入对应书籍。
            book_windows[chapter_to_book[chapter_id]] += 1  # 累计四本书实际提供的数据量。
        if step % 250 == 0 or step == expected_steps:  # 定期显示进度，最后一步也显示。
            print(f"训练 step={step}/{expected_steps} 批次损失={loss:.6f} grad_norm={raw_norm:.6f} scale={scale:.6f}", flush=True)  # 不要求不同批次损失逐行下降。
        if step in checkpoints:  # 在预定检查点记录同一固定验证子集。
            history.append(record_checkpoint(model, data.vocab, subset, step, output_dir))  # 保存当前阶段结果，不用验证梯度训练。
    elapsed = perf_counter() - started  # 结束本轮训练与周期检查点计时，尚未包含末尾完整验证。
    assert completed_steps == expected_steps and seen_windows == expected_windows, "完整一轮覆盖不正确"  # 核对没有提前停止或错误多算窗口。
    assert seen_positions == expected_windows * seq_length == int(full_counts.sum()), "训练目标计数不一致"  # 窗口统计与基线计数应覆盖相同位置。
    assert sum(book_windows.values()) == seen_windows and all(count > 0 for count in book_windows.values()), "四书覆盖不正确"  # 确认所有书籍都实际参与训练。
```

把 islice 的训练截断去掉，不是把 max_steps 随意设成很大；生成器自然结束才是一轮。统计 actual windows、actual positions 和最后批大小，避免尾批被错误补计。全训练目标计数与 RNN 看到的 y 完全一致，因此末尾全训练字频与模型使用相同目标出现范围。

每批先计算完整梯度，再裁剪并更新。每 250 步打印进度，flush=True 让长运行日志及时显示。训练批次不同，不要求日志 loss 单调下降。训练与周期检查点耗时包含中途验证、生成和文件保存，不包含初始零步检查点、数据准备及末尾完整验证，不称为纯矩阵运算速度。

### 第五块：比较固定子集与完整验证窗口集

```python
    frequency_probs = baseline.smoothed_probabilities(full_counts, alpha=1.0)  # 只根据整轮训练目标估计字频，不使用验证答案。
    subset_frequency, subset_positions = baseline.frequency_loss(frequency_probs, subset)  # 在原固定子集上重算对应的全训练集字频分数。
    assert subset_positions == history[-1]["validation_positions"], "子集比较范围不一致"  # RNN 和字频必须比较同一位置集合。
    print("开始完整验证窗口集评估（只前向，不更新参数）", flush=True)  # 提醒接下来还有完整验证计算。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 完整验证窗口按批流式提供，不全部放进内存。
    full_rnn_loss, validation_positions = training.evaluate(model, validation_batches, network)  # 固定最终模型参数，遍历完整验证窗口集。
    validation_batches = data.iter_batches(data.iter_windows(validation_records, seq_length, seq_length), batch_size)  # 新建生成器，避免复用已经耗尽的验证流。
    full_frequency_loss, frequency_positions = baseline.frequency_loss(frequency_probs, validation_batches)  # 使用同一完整验证范围评估字频基线。
    expected_validation_positions = sum(max(0, (len(record["text"]) - seq_length - 1) // seq_length + 1) for record in validation_records) * seq_length  # 按既定丢尾规则独立核对完整验证位置数。
    assert validation_positions == frequency_positions == expected_validation_positions > 0, "完整验证范围不一致"  # 两种模型必须对相同目标评估。
    print("固定验证子集 RNN/字频损失：", history[-1]["validation_loss"], subset_frequency, flush=True)  # 子集只与子集比较。
    print("完整验证窗口集 RNN/字频损失：", full_rnn_loss, full_frequency_loss, flush=True)  # 完整范围只与完整范围比较。
    print("完整验证窗口集预测位置数：", validation_positions, flush=True)  # 该数字不等于原文字符数，首字符和短尾未成为目标。
```

同一个最终模型测两个不同范围，必须分别与对应字频分数比较。不能拿完整验证 RNN 损失与旧固定子集字频损失比较，也不能把子集变成完整集合后的数值差当作模型又训练变好了。完整验证第二次使用新的批次生成器，因为第一次已经耗尽；不把 620 批全部缓存。

training.evaluate 保持参数不变；frequency_probs 只用本轮训练目标拟合，验证答案只用于查概率计算损失。未知验证字符继续映射到保存词表的未知编号，不扩充词表。测试集不在本课使用。

### 第六块：保存报告、显示统计与入口

```python
    report = {  # 记录本轮实际覆盖、固定子集历史与完整验证结果。
        "config": {  # 固定完整一轮训练与生成比较条件。
            "epochs": 1, "batch_size": batch_size, "seq_length": seq_length,  # 记录遍历轮数与批次尺寸。
            "learning_rate": learning_rate, "max_norm": max_norm, "model_seed": 42, "chapter_shuffle_seed": 42,  # 记录更新规则和随机顺序。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42,  # 记录固定生成设置。
            "validation_batches": 3, "frequency_alpha": 1.0,  # 标明中途子集范围和字频平滑。
        },  # 完整配置记录结束。
        "train_sha256": data.vocab["train_sha256"], "expected_steps": expected_steps, "completed_steps": completed_steps,  # 保存数据来源与整轮步数。
        "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": last_batch_size, "book_windows": book_windows,  # 保存真实覆盖和尾批统计。
        "training_process_loss": loss_sum / seen_positions, "training_and_checkpoint_seconds": elapsed,  # 这是参数不断变化的训练过程均值及含检查点的耗时。
        "history": history, "subset_frequency_loss": subset_frequency,  # 子集指标单独保存，不与完整验证混比。
        "full_validation": {"positions": validation_positions, "rnn_loss": full_rnn_loss, "frequency_loss": full_frequency_loss},  # 两个模型的完整验证比较。
    }  # 不加入验证损失必须下降或 RNN 必须胜出的断言。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson32.json"  # 保存独立报告，不覆盖以前课程记录。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 完整训练与验证完成后保存报告。
    print("四本书实际训练窗口数：", book_windows, flush=True)  # 显示本轮实际覆盖。
    print("整轮步数/窗口数/预测位置数/尾批行数：", completed_steps, seen_windows, seen_positions, last_batch_size, flush=True)  # 当前应为 4787、76584、2450688、8。
    print("训练过程按位置加权平均损失：", report["training_process_loss"], flush=True)  # 不把训练过程均值当作固定最终模型的训练集评估。
    print("训练与周期检查点耗时秒：", elapsed, flush=True)  # 不包含末尾完整验证，也不是纯矩阵计算耗时。
    print("报告路径：", report_path, flush=True)  # 保存的权重可独立加载，但当前脚本尚不自动断点续训。
    print("四书完整一轮、尾批与同范围字频比较检查：通过", flush=True)  # 说明流程与覆盖正确，不保证文字质量达标。


if __name__ == "__main__":  # 直接运行文件时才开始本课完整训练。
    main()  # 从初始模型完整训练一轮并评估。
```

本课完整报告在训练和完整验证完成后写入，期间检查点已经按预定步数保存。当前脚本尚未实现自动断点续训；中断后重新启动仍从头开始，已有检查点可供单独加载评估。再次完整运行会覆盖同名本课产物，旧课程权重不受影响。

### 老师整轮预跑结果

固定验证前三批的损失：

| 已完成 step | 固定子集 RNN 交叉熵 |
|---:|---:|
| 0 | 8.651848 |
| 200 | 7.676629 |
| 1000 | 6.939782 |
| 3000 | 6.698560 |
| 4787 | 6.465969 |

最终模型与全训练字频基线的同范围比较：

| 评估范围 | 目标位置数 | RNN 交叉熵 | 字频交叉熵 |
|---|---:|---:|---:|
| 原固定前三批 | 1536 | 6.465969 | 6.662431 |
| 完整验证窗口集 | 316960 | 6.163158 | 6.360865 |

当前参数在两种各自匹配的范围上都超过字频基线。这支持它在当前数据定义下取得了优于简单固定字频的预测表现，但不能据此宣称已经学会所有语法或可读性合格。最终贪心仍有“我不道”等重复，采样也不通顺，原样保留。

训练过程按位置加权损失为 6.405641，它来自一路变化的参数，不是固定最终模型的训练集评估，不能用于直接计算泛化差距。完整验证和固定子集分数也不是同一数据上的两次训练结果。

本环境训练与周期检查点约 224.9 秒，末尾完整验证另需时间，实际速度依 CPU 和 NumPy 而异。全轮仍流式处理，增加的是运行时间，不是将全部窗口展开成巨大内存张量。参数仍为 719723，单个 float32 模型裸参数数据区 2878892 字节。本课五份检查点合计 13418005 字节，各自仍远小于 1 GB；不需要在内存中同时持有五个模型。

四书完整窗口数独立核对为：三国演义 14818、水浒全传 22415、红楼梦 21299、西游记 18052，合计 76584。老师另核对前两百步与第 29 课记录一致，并独立重载最终文件生成，与记录相同；只在临时目录保存，未改仓库旧权重。最终报告字典只做排版拆行，语法树与已跑代码一致，不重复运行数分钟训练来测试纯排版。

完整参考输出：

```text
检查点 step=0 固定验证子集损失=8.651848
贪心： '春风忏帐朝瑁抅螳赋舌捶樵语捣砌张喜班貔蘅舐缨墦降莩\n擢椁迤膻休腌将隙帘姓容喟楚豂拶骈'
采样： '春风衽殆迎腹匿鱼蛮要哀汶旧隶糠贺永屉甩兼赓箔蚤摄鬃重袒姣浒保噎耻蔺骦拧料涕妆哔涸屁美'
计划整轮窗口数与更新步数： 76584 4787
检查点 step=200 固定验证子集损失=7.676629
贪心： '春风，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，'
采样： '春风袋朽锯花上，诵跋乞暮惙，筷道是匝爆。邙程诡待，髭趣僭朴。亲胭裸，岘悚来你了板南罽'
训练 step=250/4787 批次损失=7.445200 grad_norm=0.450757 scale=1.000000
训练 step=500/4787 批次损失=7.085508 grad_norm=0.400173 scale=1.000000
训练 step=750/4787 批次损失=6.165971 grad_norm=0.541877 scale=1.000000
训练 step=1000/4787 批次损失=6.769705 grad_norm=0.471205 scale=1.000000
检查点 step=1000 固定验证子集损失=6.939782
贪心： '春风，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，'
采样： '春风谢很霓缥上：西走中怎妄，甸道得入李。道玺要天，马责众懦。事等蝙，因如我令丹扎关神'
训练 step=1250/4787 批次损失=6.895018 grad_norm=0.467664 scale=1.000000
训练 step=1500/4787 批次损失=6.593189 grad_norm=0.491804 scale=1.000000
训练 step=1750/4787 批次损失=6.526593 grad_norm=0.550840 scale=1.000000
训练 step=2000/4787 批次损失=6.553858 grad_norm=0.724674 scale=1.000000
训练 step=2250/4787 批次损失=6.158853 grad_norm=0.668995 scale=1.000000
训练 step=2500/4787 批次损失=6.253382 grad_norm=1.016262 scale=0.983999
训练 step=2750/4787 批次损失=6.255238 grad_norm=1.081175 scale=0.924919
训练 step=3000/4787 批次损失=5.991628 grad_norm=0.741507 scale=1.000000
检查点 step=3000 固定验证子集损失=6.698560
贪心： '春风，，不不，，不了，，不不，，不不，，不不，，不不，，不不，，不不，，不不，，不不'
采样： '春风鏊心，百个，累蒜了文子，此见尸元来。贾殿者基，那舜伪我。一甸罡，吕大心他之方何自'
训练 step=3250/4787 批次损失=5.693136 grad_norm=0.803617 scale=1.000000
训练 step=3500/4787 批次损失=5.986103 grad_norm=0.694355 scale=1.000000
训练 step=3750/4787 批次损失=5.605667 grad_norm=0.523846 scale=1.000000
训练 step=4000/4787 批次损失=6.232615 grad_norm=1.093694 scale=0.914332
训练 step=4250/4787 批次损失=6.174207 grad_norm=0.782665 scale=1.000000
训练 step=4500/4787 批次损失=6.215238 grad_norm=0.680970 scale=1.000000
训练 step=4750/4787 批次损失=5.900957 grad_norm=0.582376 scale=1.000000
训练 step=4787/4787 批次损失=6.676804 grad_norm=0.933366 scale=1.000000
检查点 step=4787 固定验证子集损失=6.465969
贪心： '春风，不不道：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道：'
采样： '春风赐张面者三，绍脸人徒姐，此袭弟刚来。着樊羹天，这肯公怜。”爱笑鸳壕姓房你了时冉绁'
开始完整验证窗口集评估（只前向，不更新参数）
固定验证子集 RNN/字频损失： 6.465968608856201 6.662431022714794
完整验证窗口集 RNN/字频损失： 6.1631579502612155 6.36086485508896
完整验证窗口集预测位置数： 316960
四本书实际训练窗口数： {'西游记.txt': 18052, '红楼梦.txt': 21299, '水浒全传.txt': 22415, '三国演义.txt': 14818}
整轮步数/窗口数/预测位置数/尾批行数： 4787 76584 2450688 8
训练过程按位置加权平均损失： 6.405640527395872
训练与周期检查点耗时秒： 224.86163659999147
报告路径： D:\projects\playatry\docs\training_runs\lesson32.json
四书完整一轮、尾批与同范围字频比较检查：通过
```

### 实操与提交

运行 python scripts/learn/16_train_one_epoch.py。先保持本课配置，不同时改学习率、模型维度、温度或顺序。可能需要几分钟，每 250 步可看到进度。结束后提交：五个检查点子集损失、最后生成样例、尾批与总覆盖统计、完整验证 RNN/字频损失及目标位置数。报告写 docs/training_runs/lesson32.json；新权重暂被默认忽略，复查后将添加明确例外并按每课完整上传约定一起提交。

后续 CI 将优先加载已保存的检查点与报告核对，不在每次推送自动重跑整轮；具体检查范围在实操复查时记录。本课结果属于老师预跑，不提前记为学习者完成。

回答：一、尾批只有 8 个窗口、每窗 32 个目标，分数梯度应除以 256 还是 512？二、能否把完整验证窗口集的 RNN 损失与旧前三批字频损失直接比较？为什么？

### 第三十二课实操与问答复查：通过

学习者完成整轮训练并提交末尾覆盖、验证和耗时结果。老师检查实际脚本与 JSON，未修改学习者代码，也未重复执行数分钟的整轮训练。新建只读产物检查 scripts/ci/check_lesson32.py 并运行，独立加载五份保存权重，重算各自固定子集指标、最终完整验证和字频分数，全部在浮点容差内与报告一致。

实际整轮完成 4787 步，76584 个训练窗口，2450688 个目标位置，最后批次 8 行。按章节长度独立核对四书窗口数为：西游记 18052、红楼梦 21299、水浒全传 22415、三国演义 14818，与报告逐书一致。固定子集最终 RNN/字频损失为 6.465968608856201 / 6.662431022714794；完整验证 316960 个位置的 RNN/字频损失为 6.1631579502612155 / 6.36086485508896。完整验证流最后一批为一个窗口，评估均按真实位置数平均。

用户报告训练过程按位置加权均值 6.405640527395872，训练与周期检查点耗时 246.15672990004532 秒；耗时随设备和运行负载变化，不作跨平台一致性断言。它包含中途评估、生成、保存，不含末尾完整验证。训练过程均值来自不断变化的参数，不是最终固定参数下的训练集损失。

第一题回答“256”，正确：8*32=256，梯度按尾批实际位置数归一化。第二题回答“不能”，判断正确；补充原因是完整验证与原前三批的目标集合不同，必须让 RNN 与字频基线在同一集合上比较，不能把数据范围变化误认为模型优劣。

本次五份权重、JSON、脚本、模型说明与本笔记按完整上传约定纳入仓库，添加明确的 .gitignore 例外。五份权重合计 13418005 字节，每个仍远小于 1 GB。原始数据和旧检查点没有修改。

权重 SHA-256：step0000 为 630b7ea373a0d5c74544775ae4f34bcd3805ab370a77bd760e7df34c9f5848a5；step0200 为 e076bc06fe64ecedac3f7aa5cbb28be6d8f7f315f36ee10455a2a06aa50db033；step1000 为 bef13a9e6660663d191057f27b81deadb6aab5d00b201df95bc31bbad60f9cd6；step3000 为 ba74f8320d9381d4182cbea090407175f82377205e848ad23b8393f38c166527；step4787 为 7eb4e8d0aa14391c395052070672bcee974ad71af3625eebfee40769febcfb6e。

CI 只读核对本课已保存权重：检查五份配置与子集损失、生成编号合法性、逐书窗口覆盖、尾批和完整验证/字频损失。语法检查覆盖完整训练脚本，但每次推送不自动重训整轮，不把受浮点边界影响的整段生成文字当作跨平台黄金值。本地这些检查均通过。

第三十二课实操和问答通过。当前模型在两种分别匹配的验证范围上都优于字频基线，文字仍有重复，不能认定生成质量已合格。下一阶段可学习多轮训练、按轮打乱顺序与根据验证损失选择检查点；这些功能尚未实施，不提前承诺增加轮数一定改善所有指标。

## 第三十三课：多轮训练、按轮打乱与验证选择最佳检查点

状态：老师已在临时目录从已保存的一轮模型继续训练两轮，完成总三轮对照和最佳文件核对，待学习者新建 scripts/learn/17_train_epochs.py 并输入、运行和回答。未替学习者创建实际脚本，未在仓库生成本课新模型或报告。第三十二课及其他旧检查点保持不变。

### 本课范围和继续训练前提

为了避免再训练相同第一轮，本课从 lesson32_step4787.npz 加载一次训练模型，继续完整第 2、3 轮，总累计更新步数为 3*4787=14361。本脚本实际新增 9574 次更新；每轮窗口集合仍为同一 76584 个窗口，重复轮数不增加不同样本数量。隐藏状态仍在每个窗口从零开始，跨轮保留的是参数。

当前优化器是无动量普通 SGD，没有额外动量或自适应矩状态；训练没有 dropout 等随机层，在完整一轮的边界继续，下一轮顺序又由轮号种子独立决定，所以可以在当前定义下接上这份参数。这不是任意优化器或批次中途的通用自动断点续训。脚本每次重新运行仍指定从旧第一轮开始，不会自动读取本课最后进度继续到第四轮。

每轮从原章节列表重新复制，再用 shuffle_seed=42+epoch-1 打乱；第一轮为42，第二轮43，第三轮44。章内字序不变，窗口不跨章。每轮重新创建窗口/批次生成器，不能复用已耗尽的旧流。训练模型在外层 epoch 循环之前加载一次，循环中不能重新初始化或重新加载第一轮，否则会丢掉前一轮的更新。

### 最佳检查点按什么选

每轮结束都在同一完整验证窗口集（316960 个位置）上只前向评估，选择 e*=argmin_e L_val(θ_e)，候选为已经观测的第 1、2、3 轮末模型。不是训练过程损失最低，也不是最后一轮必然最好；不代表检查了每个批次的所有可能参数点。

规则为 validation_loss<best_loss 才更新，分数相等保留较早轮。例如三轮损失为6.1、6.0、6.2，应选第二轮。先把第一轮当作初始最佳，后续轮变差时仍可保留旧模型，不强迫继续训练的模型胜出。

验证集在此用于选择已经训练好的候选，这是验证集的职责；没有把验证标签加入训练计数、梯度或参数更新。它仍会间接影响模型选择，所以验证分数不是最终完全未参与选择的独立测试结果，测试集继续保留。本课不实现早停或学习率调度，仍按预定轮数全部完成。

### 第一块：新文件导入与常量

```python
from importlib import import_module  # 复用数据、模型、评估、保存和生成工具。
from time import perf_counter  # 记录本次继续训练及验证的耗时。
import json  # 读取上一轮报告，并逐轮保存新实验记录。
import random  # 为每轮建立可复现、不同的章节顺序。
import numpy as np  # 检查数值、比较权重与验证指标。

network = import_module("07_tiny_rnn")  # 复用模型结构和完整梯度裁剪。
training = import_module("11_train_steps")  # 复用加载、保存与固定参数评估。
decoding = import_module("12_generate_text")  # 复用固定设置的文字生成。
BATCH_SIZE, SEQ_LENGTH = 16, 32  # 保持批大小与窗口长度不变。
LEARNING_RATE, MAX_NORM = 0.1, 1.0  # 沿用无动量 SGD 与全局裁剪设置。
TOTAL_EPOCHS = 3  # 总共完成三轮，本脚本从已有第一轮继续完成第二、三轮。
```

这些导入只定义已有功能，不执行旧训练入口。TOTAL_EPOCHS=3 表示加上上一课的第一轮总共三轮，不是再额外训练三轮。

### 第二块：添加完整验证与报告保存函数

```python
def evaluate_full(model, data, records):  # 在完整验证窗口集上评估当前参数。
    windows = data.iter_windows(records, SEQ_LENGTH, SEQ_LENGTH)  # 每次评估都新建验证窗口流。
    batches = data.iter_batches(windows, BATCH_SIZE)  # 流式处理所有验证批次，不缓存完整分数。
    return training.evaluate(model, batches, network)  # 只前向，已有函数检查评估不改参数。


def save_report(path, report):  # 每轮完成后保存配置、指标与当前最佳选择。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读的 JSON。
```

每次验证重建完整窗口流，避免生成器耗尽；评估保持参数不变。报告每轮保存一次，状态 running 仅表示尚未完成全部计划；最后核对完最佳文件才改为 completed。

### 第三块：main 中加载一次起点并初始化最佳选择

```python
def main():  # 在完整轮边界继续训练，而不是重复初始化模型。
    run_started = perf_counter()  # 总耗时包含起始核验、继续训练、逐轮验证和生成。
    data = import_module("06_make_windows")  # 复用四书训练数据和固定词表，首次导入显示旧数据检查。
    source = json.loads((data.ROOT / "docs" / "training_runs" / "lesson32.json").read_text(encoding="utf-8"))  # 读取已经完成第一轮的报告。
    start_epoch = source["config"]["epochs"]  # 当前起点是完整训练一轮之后。
    assert start_epoch == 1 and source["train_sha256"] == data.vocab["train_sha256"], "起点轮数或数据版本不一致"  # 只接受本课预定的第一轮起点。
    assert source["config"]["batch_size"] == BATCH_SIZE and source["config"]["seq_length"] == SEQ_LENGTH and source["config"]["learning_rate"] == LEARNING_RATE and source["config"]["max_norm"] == MAX_NORM, "继续训练配置不一致"  # 保持与上一轮相同的更新规则。
    steps_per_epoch, windows_per_epoch = source["expected_steps"], source["seen_windows"]  # 使用已核验的一轮规模。
    source_row = source["history"][-1]  # 选择第一轮最后一步，不加载单窗口诊断模型。
    source_path = data.ROOT / "model" / source_row["checkpoint"]  # 旧权重只读，不覆盖。
    model, vocab, global_step = training.load_checkpoint(source_path, network)  # 只在外层循环之前加载一次，后续轮保留更新后的参数。
    assert vocab == data.vocab and model.hidden_size == 64 and global_step == start_epoch * steps_per_epoch == source["completed_steps"], "起点权重配置不一致"  # 确认模型、词表和累计步数对应第一轮结束。
    with (data.ROOT / "data" / "splits" / "validation.jsonl").open(encoding="utf-8") as file:  # 读取相同验证集，不用于参数更新。
        validation_records = [json.loads(line) for line in file]  # 所有轮都评估同一套完整验证窗口。
    initial_loss, positions = evaluate_full(model, data, validation_records)  # 独立核对加载模型的完整验证表现。
    assert positions == source["full_validation"]["positions"] and np.isclose(initial_loss, source["full_validation"]["rnn_loss"], rtol=1e-6, atol=1e-6), "起点验证结果不一致"  # 不拿旧子集分数冒充完整分数。
    output_dir = data.ROOT / "model"  # 本课使用独立的逐轮权重与最佳权重文件名。
    best_path = output_dir / "lesson33_best.npz"  # 最佳文件是独立快照，不是同一模型对象的引用。
    best_epoch, best_loss = start_epoch, initial_loss  # 第一轮也参与比较，后续训练变差时仍可保留它。
    training.save_checkpoint(best_path, model, vocab, global_step)  # 先保存起点，确保最佳文件始终对应一个已评估模型。
    history = [{"epoch": start_epoch, "global_step": global_step, "validation_loss": initial_loss, "validation_positions": positions, "checkpoint": source_path.name, "greedy": source_row["greedy"], "sampled": source_row["sampled"]}]  # 第一轮生成样例沿用对应权重的已有观察。
    report = {  # 新报告与旧第一轮报告分开保存。
        "config": {  # 固定本课从第一轮继续到第三轮的条件。
            "starting_epoch": start_epoch, "total_epochs": TOTAL_EPOCHS, "batch_size": BATCH_SIZE, "seq_length": SEQ_LENGTH,  # 记录训练起点和批次结构。
            "learning_rate": LEARNING_RATE, "max_norm": MAX_NORM, "shuffle_seed_base": 42,  # 保持更新规则并记录按轮种子起点。
            "prompt": "春风", "new_tokens": 40, "temperature": 1.0, "sampling_seed": 42,  # 生成比较条件与之前一致。
        },  # 其余结果记录在下面的历史和统计中。
        "train_sha256": data.vocab["train_sha256"], "starting_checkpoint": source_path.name, "steps_per_epoch": steps_per_epoch,  # 保留起点与数据来源。
        "history": history, "training_stats": [], "completed_epochs": start_epoch, "status": "running",  # 每完成一轮就更新进度记录。
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "best_checkpoint": best_path.name,  # 当前最佳按完整验证损失决定。
        "full_frequency_loss": source["full_validation"]["frequency_loss"],  # 沿用同范围、已核验的全训练集字频参照。
    }  # 没有加入测试集指标或自动提前停止逻辑。
    report_path = data.ROOT / "docs" / "training_runs" / "lesson33.json"  # 独立保存本课继续训练记录。
    save_report(report_path, report)  # 先保存起点，避免只有最后才有记录。
    print(f"起点 epoch={start_epoch} global_step={global_step} 完整验证损失={initial_loss:.6f}", flush=True)  # 明确本次加载的是第一轮结束模型。
    chapter_to_book = {record["id"]: record["book"] for record in data.train_records}  # 后续逐轮核对四书覆盖。
```

第一轮完整验证损失在本次重新测量并与旧报告核对；其生成样例沿用上一课对应权重的已记录观察。字频参照沿用已经核验、同一完整范围上的 6.36086485508896。反复训练相同语料不增加新的独立字符计数，不通过重复轮数再改变基线平滑强度。

best_path 保存独立磁盘快照，不使用 best_model=model 这种仍指向同一可变训练对象的做法。后续对 model 的更新不会自动改动已经保存的文件，只有验证改善才明确覆盖最佳文件。

### 第四块：main 内逐轮打乱、重建训练流并完整更新

```python
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):  # 本次依次训练第二轮和第三轮。
        records = list(data.train_records)  # 每轮从原始章节列表重新复制，不沿用上轮乱序列表。
        shuffle_seed = 42 + epoch - 1  # 第一轮为 42，第二轮为 43，第三轮为 44。
        random.Random(shuffle_seed).shuffle(records)  # 每轮可复现且通常不同，只改变章节顺序。
        batches = data.iter_batches(data.iter_windows(records, SEQ_LENGTH, SEQ_LENGTH), BATCH_SIZE)  # 每轮重新创建训练流，避免复用已耗尽的生成器。
        epoch_steps, seen_windows, seen_positions, last_batch_size = 0, 0, 0, 0  # 这些覆盖计数每轮从零开始。
        loss_sum = 0.0  # 每轮单独统计训练过程损失。
        book_windows = {record["book"]: 0 for record in records}  # 每轮都应完整访问四本书。
        started = perf_counter()  # 此计时只覆盖本轮训练与进度输出，不含轮末验证和保存。
        print(f"开始 epoch={epoch} 章节打乱种子={shuffle_seed} 前三章={[record['id'] for record in records[:3]]}", flush=True)  # 展示本轮顺序已经重建。
        for batch in batches:  # 完整处理本轮所有窗口，包括八行尾批。
            x, y = batch["x"], batch["y"]  # 当前批次实际输入与目标。
            loss, raw_grads = model.loss_and_grad(x, y)  # 使用上一轮留下的参数继续求梯度，不重新初始化。
            assert np.isfinite(loss), "训练损失异常"  # 无效损失不能继续更新。
            grads, _, _ = network.clip_gradients(raw_grads, MAX_NORM)  # 完整反向后统一裁剪。
            for name in model.params:  # 用本批完整梯度更新五组参数。
                model.params[name] -= LEARNING_RATE * grads[name]  # 下一批及下一轮都继续使用这些更新。
            assert all(np.isfinite(param).all() for param in model.params.values()), "模型参数异常"  # 及时发现训练中的无效数值。
            epoch_steps += 1  # 本轮步数只统计当前轮。
            global_step += 1  # 累计步数不在轮与轮之间清零。
            seen_windows += x.shape[0]  # 按实际批大小累加窗口。
            seen_positions += y.size  # 按实际预测位置数统计。
            last_batch_size = x.shape[0]  # 记录本轮尾批行数。
            loss_sum += loss * y.size  # 训练过程损失按实际位置数加权。
            for chapter_id in batch["chapter_ids"]:  # 将每行窗口计入对应书籍。
                book_windows[chapter_to_book[chapter_id]] += 1  # 各轮窗口数量应相同，顺序可以不同。
            if epoch_steps % 500 == 0 or epoch_steps == steps_per_epoch:  # 定期输出本轮与累计进度。
                print(f"epoch={epoch} step={epoch_steps}/{steps_per_epoch} global_step={global_step} loss={loss:.6f}", flush=True)  # 不要求不同批次损失单调下降。
        epoch_seconds = perf_counter() - started  # 训练计时结束，接下来保持参数固定评估。
        assert epoch_steps == steps_per_epoch and global_step == epoch * steps_per_epoch and seen_windows == windows_per_epoch, "轮数或更新覆盖错误"  # 核对没有漏掉或重复完整轮。
        assert seen_positions == windows_per_epoch * SEQ_LENGTH and last_batch_size == source["last_batch_size"] and book_windows == source["book_windows"], "本轮数据覆盖不一致"  # 顺序变化不能改变窗口集合与尾批大小。
```

epoch_steps、每轮覆盖计数与过程损失在各轮开始清零；model 和 global_step 不能清零。第2轮global_step从4787增加到9574，第3轮再增加到14361；每轮仍保留八行尾批。按书计数必须与第一轮完整窗口集合一致，只改变访问顺序。

每轮 training_seconds 只覆盖该轮训练及进度输出，不含轮末验证、生成和保存；它与第三十二课包含周期检查点的计时口径不同，不能直接拿来当作同等操作速度比较。

### 第五块：轮末评估、保留全部历史并按条件更新最佳

```python
        checkpoint_path = output_dir / f"lesson33_epoch{epoch:02d}.npz"  # 每轮单独保存，不覆盖旧轮文件。
        training.save_checkpoint(checkpoint_path, model, vocab, global_step)  # 元数据记录累计更新步数，不是当前轮内步数。
        validation_loss, current_positions = evaluate_full(model, data, validation_records)  # 所有轮都用完整同一验证窗口集。
        assert current_positions == positions, "验证范围在轮之间改变"  # 不将不同范围的损失用于最佳模型选择。
        greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的贪心生成设置。
        sampled, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32, temperature=1.0, seed=42)  # 固定温度和采样种子。
        history.append({"epoch": epoch, "global_step": global_step, "validation_loss": validation_loss, "validation_positions": current_positions, "checkpoint": checkpoint_path.name, "greedy": greedy, "sampled": sampled})  # 每轮结果都保留，不只留下最好的一轮。
        report["training_stats"].append({"epoch": epoch, "shuffle_seed": shuffle_seed, "first_chapter_ids": [record["id"] for record in records[:3]], "epoch_steps": epoch_steps, "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": last_batch_size, "book_windows": book_windows, "training_process_loss": loss_sum / seen_positions, "training_seconds": epoch_seconds})  # 记录顺序、覆盖与训练统计。
        if validation_loss < best_loss:  # 只有完整验证损失严格更低才替换最佳，相等时保留较早轮。
            best_epoch, best_loss = epoch, validation_loss  # 更新最佳轮数及对应分数。
            training.save_checkpoint(best_path, model, vocab, global_step)  # 保存独立文件快照，后续训练不会继续修改它。
        report["best_epoch"], report["best_validation_loss"] = best_epoch, best_loss  # 报告同步记录当前选择。
        report["completed_epochs"] = epoch  # 该轮训练和验证完成后才记录为完成。
        save_report(report_path, report)  # 每轮落盘一次，即使后面中断仍保留此前完整轮的记录。
        print(f"轮末 epoch={epoch} 完整验证损失={validation_loss:.6f} 当前最佳轮={best_epoch} 最佳损失={best_loss:.6f}", flush=True)  # 最后轮与最佳轮可能不同。
        print("贪心：", repr(greedy), flush=True)  # 如实输出重复或不通顺。
        print("采样：", repr(sampled), flush=True)  # 固定设置下记录文字变化。
```

每轮文件为 lesson33_epoch02.npz、lesson33_epoch03.npz，最佳别名为 lesson33_best.npz。即使某轮不是最佳，也保存它和对应指标，不能只留好看的记录。最佳文件的 metadata 记录其累计更新步数，而非当前轮内步数。

### 第六块：训练结束，独立加载最佳文件核对

```python
    selected = min(history, key=lambda row: row["validation_loss"])  # 独立从所有已评估轮末记录核对最低完整验证损失。
    assert selected["epoch"] == best_epoch and np.isclose(selected["validation_loss"], best_loss), "最佳选择与记录不一致"  # 不把最后一轮直接当作最好。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后重新读取实际最佳文件。
    selected_path = source_path if selected["epoch"] == start_epoch else output_dir / selected["checkpoint"]  # 最佳可能仍是旧第一轮，也可能是新增轮。
    selected_model, _, selected_step = training.load_checkpoint(selected_path, network)  # 加载被选中轮的独立检查点作对照。
    assert best_vocab == vocab and best_step == selected_step == selected["global_step"] and all(np.array_equal(param, selected_model.params[name]) for name, param in best_model.params.items()), "最佳文件与所选轮参数不一致"  # 防止最佳文件其实保存了最后一轮或可变引用。
    best_text, _ = decoding.generate(best_model, best_vocab, "春风", new_tokens=40, context_size=32)  # 结果展示使用最佳文件加载的模型。
    report["status"], report["run_seconds"] = "completed", perf_counter() - run_started  # 只有全部训练和最佳文件核对完成才标记结束。
    save_report(report_path, report)  # 保存最终完成状态与耗时。
    print("总完成轮数与累计更新步数：", report["completed_epochs"], global_step, flush=True)  # 当前应为三轮、14361 步，其中第一轮来自上一课。
    print("选中最佳轮与完整验证损失：", best_epoch, best_loss, flush=True)  # 该选择只在本课观察过的三个轮末之间比较。
    print("同范围字频基线损失：", report["full_frequency_loss"], flush=True)  # 重复遍历同一训练语料不增加独立样本。
    print("加载最佳文件后的贪心：", repr(best_text), flush=True)  # 不把文字样例替代完整验证指标。
    print("最佳权重与报告：", best_path, report_path, flush=True)  # 旧第一轮文件保持不变。
    print("多轮覆盖、逐轮打乱与最佳快照检查：通过", flush=True)  # 不保证增加轮数必然提升可读性。


if __name__ == "__main__":  # 直接运行脚本时才继续训练本课两轮。
    main()  # 从已完成一轮的模型继续到总共三轮。
```

训练模型始终是循环外加载的同一个对象；最后再次加载的是用于核对与展示的独立对象，不是在训练中重置参数。比较最佳文件和被选中的逐轮文件的五组参数完全一致，防止“报告选第二轮，文件其实是第三轮”这类错误。

### 老师完整预跑结果

| 轮末 epoch | 累计更新步数 | 完整验证窗口交叉熵 |
|---:|---:|---:|
| 1 | 4787 | 6.163158 |
| 2 | 9574 | 6.114650 |
| 3 | 14361 | 6.011733 |

当前预跑选中第 3 轮，完整验证损失 6.011733，低于同范围字频基线 6.360865。这次最佳恰好是最后一轮，但选择规则不能写成“总用最后一轮”；以后验证变差时应保留更早的模型。

贪心和采样仍有明显重复及不通顺，验证交叉熵继续改善不等于生成可读性已经达标。本次没有用生成样例临时调整温度、学习率或轮数来追求好看的输出。

两轮新增训练每轮均完整覆盖76584个窗口、2450688个目标，四书窗口数与第32课一致；独立检查了累计步数、各轮打乱种子43/44，以及最佳文件与所选轮的参数相同。老师预跑总耗时（含起点核对、两轮训练、验证、生成和保存）约 501.2 秒；实际取决于 CPU 与 NumPy，不作运行时间承诺。模型仍为719723个参数，本课三个新文件合计 8063524 字节；最佳文件可能与某份逐轮文件参数完全相同，这是明确快照而非新增模型容量。

预跑只写临时目录；最终配置字典仅为易读排版拆行，语法树与已运行源码一致，不再重复运行数分钟训练来测试纯排版。完整参考输出：

```text
起点 epoch=1 global_step=4787 完整验证损失=6.163158
开始 epoch=2 章节打乱种子=43 前三章=['三国演义:034', '水浒全传:062', '西游记:067']
epoch=2 step=500/4787 global_step=5287 loss=6.011188
epoch=2 step=1000/4787 global_step=5787 loss=5.904659
epoch=2 step=1500/4787 global_step=6287 loss=6.079937
epoch=2 step=2000/4787 global_step=6787 loss=6.644821
epoch=2 step=2500/4787 global_step=7287 loss=5.935474
epoch=2 step=3000/4787 global_step=7787 loss=5.570732
epoch=2 step=3500/4787 global_step=8287 loss=6.057013
epoch=2 step=4000/4787 global_step=8787 loss=6.062494
epoch=2 step=4500/4787 global_step=9287 loss=5.511077
epoch=2 step=4787/4787 global_step=9574 loss=5.738286
轮末 epoch=2 完整验证损失=6.114650 当前最佳轮=2 最佳损失=6.114650
贪心： '春风，不曰：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道：“'
采样： '春风逡徐，物之，筛藏五径妹，梁行往别河。罢棒维奉，跷自元念。”犼秘，噇套恩使。十即火'
开始 epoch=3 章节打乱种子=44 前三章=['三国演义:071', '三国演义:066', '三国演义:025']
epoch=3 step=500/4787 global_step=10074 loss=7.080385
epoch=3 step=1000/4787 global_step=10574 loss=6.245220
epoch=3 step=1500/4787 global_step=11074 loss=5.800443
epoch=3 step=2000/4787 global_step=11574 loss=6.299842
epoch=3 step=2500/4787 global_step=12074 loss=6.226132
epoch=3 step=3000/4787 global_step=12574 loss=5.555647
epoch=3 step=3500/4787 global_step=13074 loss=6.107298
epoch=3 step=4000/4787 global_step=13574 loss=5.635727
epoch=3 step=4500/4787 global_step=14074 loss=5.330650
epoch=3 step=4787/4787 global_step=14361 loss=6.159735
轮末 epoch=3 完整验证损失=6.011733 当前最佳轮=3 最佳损失=6.011733
贪心： '春风，不是道：“你不玉，一个，不是，一个，不是，一个，不是，一个，不是，一个，不是，'
采样： '春风都弄非虫。锋竟自二归姑，是要平出桌。玄植老头，贾红便得。”王着，只妇得住个我公自'
总完成轮数与累计更新步数： 3 14361
选中最佳轮与完整验证损失： 3 6.011732826733577
同范围字频基线损失： 6.36086485508896
加载最佳文件后的贪心： '春风，不是道：“你不玉，一个，不是，一个，不是，一个，不是，一个，不是，一个，不是，'
最佳权重与报告： D:\projects\playatry\model\lesson33_best.npz D:\projects\playatry\docs\training_runs\lesson33.json
多轮覆盖、逐轮打乱与最佳快照检查：通过
```

### 实操与提交

运行 python scripts/learn/17_train_epochs.py。需要先已有第32课完整报告和一轮权重；不要把单窗口诊断权重替换进去。运行可能需要数分钟，按每500步查看进度；中断后可保留此前完整轮的报告和文件，但重新运行默认仍从第32课的一轮模型开始，不自动接上中断点。

本课将生成 model/lesson33_epoch02.npz、model/lesson33_epoch03.npz、model/lesson33_best.npz 和 docs/training_runs/lesson33.json。权重暂按现有规则忽略，完成后复查、添加明确例外、配置只读检查并完整上传，CI 不会每次重跑两轮训练。

请提交第2、3轮完整验证损失、最后最佳轮数与分数、累计步数和最佳模型生成结果。回答：一、每轮开始是否应该重新创建或加载初始训练模型？二、如果第2轮验证损失6.0，第3轮6.2，应保留哪一轮作为最佳，为什么？本课仍是待学习者完成的练习，不提前记录其成功。

### 学习者实操验收与答题记录

学习者已完成第三十三课，实际报告为 `docs/training_runs/lesson33.json`。逐轮完整验证损失：第一轮 6.1631579502612155，第二轮 6.114649504065093，第三轮 6.011732826733577；同范围字频基线 6.36086485508896。累计更新 14,361 步，每个新增轮均覆盖 76,584 个窗口、2,450,688 个预测位置，尾批 8 行；按章打乱种子为 43、44。实际总耗时约 594.3 秒，第二、三轮纯训练耗时约 334.4、215.3 秒，计时范围与上课不同，不能直接据此推断训练效率变化。

回答一“每轮开始不应该重新创建或加载初始训练模型”正确：同一次连续训练保留上一轮参数。允许在开始继续训练时加载上一轮检查点，本课只在外层循环之前加载一次。回答二“应选第二轮作为最佳，因为损失更小”正确：这针对题设第二轮 6.0、第三轮 6.2；本次实际第三轮更低，因此实际选择第三轮，两者并不矛盾。

第三轮比第一轮完整验证交叉熵降低约 0.151425，比字频基线低约 0.349132；这是按预测位置平均的自然对数损失差，不能读成准确率提高多少百分点。贪心仍出现“一个，不是”的循环，采样仍不通顺；评价指标改善与可读性仍需分别观察。多轮重复访问的是同一组训练章节，不增加独立语料。

复查方式为审阅实际脚本，独立重建每轮章节顺序及应有覆盖，加载三个轮末文件重算完整验证损失，并将最佳文件与被选中轮的五组参数逐项比较。只读检查纳入 `scripts/ci/check_lesson33.py` 和 GitHub Actions；不为复查重复训练两轮。仅清理学习者脚本末尾空白，已通过语法树比较确认计算逻辑不变。代码、报告、三个权重及说明按每课完整上传约定同步到仓库。

## 第三十四课：切断循环连接，检查前文信息是否帮助预测

本课承接第三十三课最佳检查点，学习消融实验。学习者尚未实操完成，不提前记录验收成功。目标是回答：正常 RNN 的前文通路对当前验证预测是否有帮助？先做一次受控诊断，再决定后续优化方向。本课不训练、不新建词表，不保存新权重。

正常模型：h_t = tanh(E[x_t] + h_{t-1} R + b)，z_t = h_t W + c。独立副本只把 R 置零：h_t = tanh(E[x_t] + b)。因此副本仍知道当前输入字符 x_t，可以利用当前字预测下一字；它不再通过隐藏状态读取更早字符。它既不是均匀预测，也不是只按全局字频预测。

首位置的 h_0 原本就是零，所以两者首位置分数相同；后续位置通常不同。这也说明本课原模型只有窗口内部最长三十二个输入位置的上下文，并未读取完整章节之前的所有文字。

损失差定义为 ΔL = L_切断 - L_正常。若为正，说明在固定其余参数及本验证范围的条件下，移除这条通路使预测变差。若为负，则应记录切断后改善并继续排查，不能为了迎合预期修改断言。这个实验不能证明模型理解语法，也不能说明独立重新训练的无循环模型一定更差：其余参数原本与循环连接共同训练，突然切断也会带来分布变化。

新建 `scripts/learn/18_context_ablation.py`，按下面四块依次手输。第一块在文件顶层；第二块定义 main；第三块继续 main，保持四格缩进；第四块回到文件顶层。每个非空代码行都有直白注释。

### 第一块：复用已有评估工具

```python
from importlib import import_module  # 复用模型、权重加载与验证函数。
from pathlib import Path  # 从脚本位置确定项目根目录。
import json  # 读取上一课报告并保存本课实验记录。
import numpy as np  # 检查参数独立性和数值结果。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn，向上两级得到根目录。
network = import_module("07_tiny_rnn")  # 导入已经完成的 RNN。
training = import_module("11_train_steps")  # 复用权重加载与只前向评估。


def evaluate_full(model, data, records):  # 每次使用相同的完整验证窗口。
    windows = data.iter_windows(records, 32, 32)  # 创建新的窗口流，避免生成器已经耗尽。
    batches = data.iter_batches(windows, 16)  # 使用十六行批次并保留尾批。
    return training.evaluate(model, batches, network)  # 按实际位置加权，并检查参数没有变化。
```

### 第二块：加载两个独立模型，只改副本的循环矩阵

```python
def main():  # 比较原模型与切断循环连接的独立副本。
    data = import_module("06_make_windows")  # 复用训练词表和相同窗口规则，导入时会显示旧数据检查。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 读取上一课选出的最佳模型信息。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点未完成或数据版本不一致"  # 不接受错误来源的模型。
    path = ROOT / "model" / source["best_checkpoint"]  # 始终使用上一课验证选出的最佳权重。
    original, vocab, step = training.load_checkpoint(path, network)  # 第一次加载原模型。
    ablated, other_vocab, other_step = training.load_checkpoint(path, network)  # 第二次加载独立副本，不能写成 ablated = original。
    assert vocab == other_vocab == data.vocab and step == other_step, "模型配置不一致"  # 两个模型必须从同一检查点开始。
    assert all(np.array_equal(value, ablated.params[name]) and not np.shares_memory(value, ablated.params[name]) for name, value in original.params.items()), "模型参数不相同或共享内存"  # 相同数值也必须存放在独立数组中。
    saved = {name: value.copy() for name, value in original.params.items()}  # 留下原模型参数，结束时核对它没有变化。
    ablated.params["recurrent"].fill(0.0)  # 只在副本中把循环矩阵置零，切断来自前一步状态的贡献。
    assert all(np.array_equal(value, saved[name]) for name, value in ablated.params.items() if name != "recurrent"), "其他参数被改变"  # 表示表、偏移与输出层保持原值。
    records = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 两种条件使用相同验证章节。
    preview = next(data.iter_batches(data.iter_windows(records, 32, 32), 16))  # 取一个批次检查切断连接的直接影响。
    original_logits = original.forward(preview["x"])  # 原模型在这个批次上的全部分数。
    ablated_logits = ablated.forward(preview["x"])  # 副本在相同输入上的全部分数。
    assert np.allclose(original_logits[:, 0, :], ablated_logits[:, 0, :]), "首位置分数应相同"  # 每个窗口的初始状态为零，首位置原本也没有前文贡献。
    later_difference = float(np.max(np.abs(original_logits[:, 1:, :] - ablated_logits[:, 1:, :])))  # 测量后续位置分数的最大变化。
    assert later_difference > 0, "后续位置分数没有变化"  # 确认这次干预确实影响了递推计算。
```

### 第三块：比较完整验证损失并检查原模型

```python
    original_loss, positions = evaluate_full(original, data, records)  # 在完整验证窗口上测量正常模型。
    print("正常模型完整验证损失：", original_loss, flush=True)  # 先显示第一个条件的结果。
    ablated_loss, other_positions = evaluate_full(ablated, data, records)  # 同样位置上测量切断连接后的副本。
    assert positions == other_positions == 316960, "验证范围不一致"  # 两个平均数必须覆盖相同预测位置。
    assert np.isclose(original_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "正常模型与上课指标不一致"  # 先确认没有加载错文件或改变验证范围。
    assert all(np.array_equal(value, saved[name]) for name, value in original.params.items()), "原模型被修改"  # 副本干预和全部评估都不能改变原模型。
    delta = ablated_loss - original_loss  # 正数表示切断前文连接后损失上升。
    report = {  # 保存实验条件与结果，不保存干预后的模型权重。
        "checkpoint": path.name, "completed_steps": step, "train_sha256": vocab["train_sha256"],  # 记录模型和数据来源。
        "config": {"batch_size": 16, "seq_length": 32, "intervention": "zero_recurrent_without_retraining"},  # 明确这是不重新训练的干预。
        "positions": positions, "original_loss": original_loss, "ablated_loss": ablated_loss,  # 两种条件都覆盖完整验证窗口集。
        "loss_increase": delta, "full_frequency_loss": source["full_frequency_loss"],  # 字频是另一个已独立拟合的参照，不能与副本混为一谈。
        "preview_later_max_logit_difference": later_difference,  # 保留预览批次后续位置确实改变的证据。
    }  # 这份报告不使用测试集，也没有参数更新。
    print("切断循环连接后的损失：", ablated_loss, flush=True)  # 如实显示结果，不预先断言必须更差。
    print("切断后损失增加量：", delta, flush=True)  # 比较按位置平均的自然对数损失差。
    print("同范围字频基线：", report["full_frequency_loss"], flush=True)  # 保留上一课已经核对的参照值。
    print("验证预测位置数：", positions, flush=True)  # 明确当前分数的覆盖规模。
    print("独立副本、首位置相同与原模型保留检查：通过", flush=True)  # 检查计算和控制条件，而非断言实验结论。
    return report  # 返回普通字典，交给入口保存。
```

### 第四块：运行并保存报告

```python
if __name__ == "__main__":  # 只有直接运行本脚本时才执行实验并保存报告。
    result = main()  # 运行两个条件的完整验证，不训练模型。
    report_path = ROOT / "docs/training_runs/lesson34.json"  # 把本课结果放入独立记录文件。
    report_path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可读的中文 JSON。
    print("实验报告：", report_path)  # 告诉你检查结果文件的位置。
```

### 操作与思考

运行 `python scripts/learn/18_context_ablation.py`。两个条件各遍历一次完整验证窗口集，可能需要几十秒，原模型权重文件不变。把两个损失、增加量、验证位置数贴给老师；检查后再将本课脚本及报告完整提交 GitHub。

问题一：循环矩阵置零以后，副本还知道当前输入字吗？问题二：若正常损失 6.0、切断后 6.8，这能否证明模型已经掌握语法？先回答这两个小问题，不需要提前推导新的梯度。

### 老师只读预跑

仅在内存执行上面的同一份参考代码，没有替学习者创建练习脚本或正式实验报告。完整验证位置 316960，正常损失 6.011732826733577，切断后损失 6.1102805177591355，增加量 0.09854769102555849。两个条件评估后原参数不变、首位置分数一致、后续分数存在差异的检查通过。此结果支持该模型在这个验证范围内受益于循环通路，不代表语言质量已达标。

### 学习者实操验收与概念补充

学习者已运行第三十四课，正式报告 `docs/training_runs/lesson34.json`：正常损失 6.011732826733577，切断循环连接后 6.1102805177591355，增加 0.09854769102555849；两个条件均覆盖 316960 个验证预测位置。预览批次后续位置最大分数差为 5.447765350341797。学习者脚本与参考计算一致，仅清理末尾空白，语法树对比确认逻辑不变。

第1题学习者回答“不知道”，该概念仍需巩固：R 置零只删除 h_{t-1}R，当前字的 E[x_t] 仍在。以输入“春风”预测下一字为例，正常模型可结合“春”和“风”；副本计算最后位置时仍读到“风”，但无法沿状态通路带入“春”。因此副本可能预测“风”后常见的字，不等同于完全不看输入的全局字频基线。输入编号、表示表、隐藏偏移和输出层都没有被删除。

第2题回答“不能证明”正确：当前结论只支持该检查点在本验证范围内受益于循环通路。突然切断共同训练的连接并不是重新训练的无循环模型比较，也不能据此证明掌握语法或长距离依赖。正常窗口内部最长32个输入位置，状态不会跨窗口保留。

下一次概念确认只问一个具体问题：“如果输入改成‘秋风’，切断循环连接的副本在最后位置计算时，还能知道前面的‘秋’吗？”预期答案为不能；末位置只使用当前‘风’的向量及保留的固定参数。在固定权重、无随机前向的当前实现中，‘春风’与‘秋风’副本末位置分数相同。尚未收到该确认，不记录为已掌握。

复查通过后将代码、正式报告、笔记与只读检查完整上传。`scripts/ci/check_lesson34.py` 调用 main 复算两个条件并核对报告，在浮点容差内比较；同时检查最佳权重文件 SHA-256 和正式报告字节未变。CI 不执行报告保存入口，不训练或新增权重。

### 第三十四课后续理解确认

学习者先误答切断循环后还能知道“秋风”中的“秋”；重新用公式解释仅保留 E[风]+b 后，对“春风”和“秋风”最后位置预测分数是否相同回答“相同”，该具体例子已答对。这里记录的是这一例子的理解确认，不等同于掌握所有循环网络概念。

## 第三十五课：困惑度与逐字准确率

本课只评估第三十三课最佳模型，继续对照相同完整验证范围的训练字频基线。学习者尚未完成本课，不预记验收。新增文件由学习者手输：scripts/learn/19_evaluation_metrics.py；结果写入 docs/training_runs/lesson35.json，不保存权重或使用测试集。

交叉熵 L = -(1/N) Σ ln p_i，其中 p_i 是第 i 个位置正确答案的预测概率。困惑度 PPL = exp(L) = 1 / (Π p_i)^(1/N)，即正确答案概率几何平均值的倒数。均匀分布在 K 个候选上时 PPL=K，这提供直觉，但一般情况不表示模型真的在固定 K 个字中选择，也不等于错误字数。更不能用 1/PPL 作为准确率；它对应几何平均正确答案概率。对同一范围和编码方式，PPL 是 L 的单调变换，因此排序相同，不是新增独立证据。

准确率 = 最高分编号与正确编号相同的位置数 / 总位置数。logits 和 softmax 概率的最大项编号相同，因此直接对 logits 最后一维 argmax。若 logits 为 (16,32,5547)，预测编号为 (16,32)，与 y 比较后为同形状布尔数组，count_nonzero 统计 True 的数量。

举例：正确答案都是“风”，两次分布均以“风”为最高项，其概率分别0.6、0.9，则两次 top1 都命中，准确率均为100%，但第二次交叉熵更低。准确率只看最高项是否正确，交叉熵看给正确答案分配多少概率。预测仍使用真实前文的 teacher forcing，并非检验自由生成的一整段文字。全部编码位置包括标点、换行与 UNK，不能称为仅汉字准确率。

统计规则：累加所有位置损失后除以总位置数，再取 exp；不能直接平均各批 PPL。准确率同样用总命中数除以总位置数，不能把大批和尾批的准确率等权平均。这里用 float64 累加已算出的逐位损失，和先对 float32 批内损失取均值的旧报告可能有微小舍入差，核对使用1e-6容差。

按下面四块依次输入：第一块为顶层导入和汇总函数，第二块定义 main，第三块继续 main 保持四格缩进，第四块回到顶层入口。

### 准备指标汇总函数

```python
from importlib import import_module  # 复用已有数据、模型和字频工具。
from pathlib import Path  # 根据当前脚本位置找到项目目录。
import json  # 读取上一课记录，并保存新的评估报告。
import numpy as np  # 计算预测编号、自然指数和统计指标。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于 scripts/learn，向上两级得到根目录。
network = import_module("07_tiny_rnn")  # 使用已经完成的模型结构与交叉熵函数。
training = import_module("11_train_steps")  # 复用检查点加载函数。
baseline = import_module("14_frequency_baseline")  # 复用训练集计数的平滑概率函数。


def summarize(loss_sum, correct, positions):  # 将全体位置的累计值转换为评估指标。
    assert positions > 0 and 0 <= correct <= positions, "评估数量不合法"  # 避免空数据及错误命中数。
    loss = loss_sum / positions  # 先求所有位置的平均交叉熵。
    perplexity = float(np.exp(loss))  # 困惑度等于自然指数的平均交叉熵次方。
    assert np.isfinite(loss) and loss >= 0 and np.isfinite(perplexity), "指标不是有效有限数"  # 发现无效计算时及时停止。
    return {"loss": loss, "perplexity": perplexity, "correct": correct, "accuracy": correct / positions}  # 准确率保存为零到一之间的小数。
```

### 加载模型并重建训练字频基线

```python
def main():  # 在相同完整验证位置比较 RNN 和字频基线。
    data = import_module("06_make_windows")  # 复用既有词表与窗口，首次导入会打印旧数据检查。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 读取验证选出的最佳检查点信息。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 防止不同实验数据混用。
    model, vocab, step = training.load_checkpoint(ROOT / "model" / source["best_checkpoint"], network)  # 加载已完成训练的最佳模型。
    assert vocab == data.vocab, "模型词表与窗口编码不一致"  # 相同编号必须对应相同字符。
    saved = {name: value.copy() for name, value in model.params.items()}  # 评估结束后检查参数未变。
    counts = np.zeros(model.vocab_size, dtype=np.int64)  # 给每个目标编号准备训练计数。
    for batch in data.iter_batches(data.iter_windows(data.train_records, 32, 32), 16):  # 只从训练窗口统计目标字符。
        counts += np.bincount(batch["y"].reshape(-1), minlength=model.vocab_size)  # 累加每个编号的实际出现次数。
    assert int(counts.sum()) == 2450688, "训练目标计数范围不一致"  # 与之前字频基线使用相同目标集合。
    probabilities = baseline.smoothed_probabilities(counts, alpha=1.0)  # 沿用预先固定的加一平滑。
    log_probabilities = np.log(probabilities)  # 字频分布不随输入变化，只需计算一次对数。
    frequency_prediction = int(np.argmax(probabilities))  # 字频基线在每个位置都选同一个最高频编号。
    records = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 验证文本只用于测量。
    rnn_loss_sum, frequency_loss_sum = 0.0, 0.0  # 分别累计两个模型的逐位置损失。
    rnn_correct, frequency_correct, positions = 0, 0, 0  # 命中数与位置数都按实际数量统计。
```

### 同一验证流上统计损失及命中数

```python
    for batch in data.iter_batches(data.iter_windows(records, 32, 32), 16):  # 两种预测在完全相同的批次上评估。
        x, y = batch["x"], batch["y"]  # 读取输入及每个位置正确的下一字编号。
        logits = model.forward(x)  # 只前向计算，不求梯度或更新参数。
        _, position_losses = network.cross_entropy(logits, y)  # 保留每个预测位置的交叉熵。
        predictions = np.argmax(logits, axis=-1)  # 对词表维度取最高分，得到与 y 相同的二维形状。
        rnn_loss_sum += float(position_losses.sum(dtype=np.float64))  # 双精度累计逐位损失，避免把尾批等权处理。
        frequency_loss_sum += float((-log_probabilities[y]).sum(dtype=np.float64))  # 同样统计字频基线正确答案的负对数概率。
        rnn_correct += int(np.count_nonzero(predictions == y))  # 统计模型最高分编号等于正确答案的数量。
        frequency_correct += int(np.count_nonzero(y == frequency_prediction))  # 统计字频基线的命中位置数量。
        positions += y.size  # 尾批只有实际存在的位置参与评估。
    assert positions == 316960, "完整验证位置数不正确"  # 本课仍使用相同完整验证窗口集。
    rnn_metrics = summarize(rnn_loss_sum, rnn_correct, positions)  # 先汇总全部位置，再计算困惑度。
    frequency_metrics = summarize(frequency_loss_sum, frequency_correct, positions)  # 对基线使用完全相同的公式。
    assert np.isclose(rnn_metrics["loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "RNN 损失与旧记录不一致"  # 累计精度不同可能产生微小浮点差。
    assert np.isclose(frequency_metrics["loss"], source["full_frequency_loss"], rtol=1e-6, atol=1e-6), "字频损失与旧记录不一致"  # 确认概率拟合和评估范围没有改变。
    assert all(np.array_equal(value, saved[name]) for name, value in model.params.items()), "评估改变了参数"  # 本课全程固定模型。
    report = {  # 保存可复查的评估条件、命中数及指标。
        "checkpoint": source["best_checkpoint"], "completed_steps": step, "train_sha256": vocab["train_sha256"],  # 记录来源。
        "config": {"batch_size": 16, "seq_length": 32, "frequency_alpha": 1.0, "log_base": "e", "accuracy": "top1_all_encoded_positions"},  # 明确自然对数及全部编码位置的准确率。
        "positions": positions, "frequency_prediction_token": vocab["tokens"][frequency_prediction],  # 保留分母和基线恒定预测字符。
        "rnn": rnn_metrics, "frequency": frequency_metrics,  # 两组结果使用相同词表和验证范围。
    }  # 未知编号、标点和换行也计入本次全部位置评估。
    for name in ("rnn", "frequency"):  # 逐项显示两个模型的结果。
        metrics = report[name]  # 当前模型的一组指标。
        print(f"{name} loss={metrics['loss']:.6f} perplexity={metrics['perplexity']:.3f} accuracy={metrics['accuracy']:.2%} correct={metrics['correct']}/{positions}", flush=True)  # 准确率仅在显示时转成百分比。
    print("同范围损失、困惑度、准确率与参数保留检查：通过", flush=True)  # 指标正确不代表生成文字已经流畅。
    return report  # 返回结果，便于检查工具只读复算。
```

### 运行并保存报告

```python
if __name__ == "__main__":  # 直接运行脚本时才保存本课报告。
    result = main()  # 只评估已有模型，不重新训练。
    path = ROOT / "docs/training_runs/lesson35.json"  # 本课使用独立的评估记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存 JSON，保留中文字符。
    print("评估报告：", path)  # 输出结果文件位置。
```

### 实操任务

运行 python scripts/learn/19_evaluation_metrics.py，贴出末尾两组 loss、perplexity、accuracy、correct/positions。思考一：交叉熵为 ln(4) 时，困惑度是多少？思考二：正确答案概率从0.6变为0.9且两次它都为最高项，准确率会变化吗？交叉熵会变化吗？完成检查后再完整上传代码及正式评估报告，CI只读复算。

### 老师预跑结果

只在内存执行参考代码，没有创建学习者练习脚本或正式报告。两个条件使用 316960 个完整验证位置，字频最高项为 '，'。

| 模型 | 交叉熵 | 困惑度 | 命中数 | 准确率 |
|---|---:|---:|---:|---:|
| rnn | 6.011732811 | 408.190024 | 39502 | 12.462771% |
| frequency | 6.360864855 | 578.746672 | 22927 | 7.233405% |

参数保留、同范围旧损失对照检查均通过。模型优于全局字频，但上述指标都不能替代自由生成质量检查。

### 学习者第三十五课实操验收与回答

学习者已提交脚本与正式报告 docs/training_runs/lesson35.json。完整验证316960个位置，RNN交叉熵6.011732810694507、困惑度408.1900238440266、命中39502、准确率0.12462771327612317；字频交叉熵6.36086485508896、困惑度578.7466720167972、命中22927、准确率0.07233404846037354。RNN准确率比字频高约5.23个百分点，这不同于相对提高5.23%。两者均使用给定真实前文的逐位置评估，不能解释为自由生成的文字有12.46%正确。

第1题回答 PPL=4 正确，因为 exp(ln4)=4。第2题回答“准确率不变，交叉熵变”正确，补充变化方向：两次正确答案均为最高项，命中情况不变；正确答案概率从0.6增加到0.9，单位置交叉熵从-ln(0.6)约0.510826降至-ln(0.9)约0.105361。模型可以在准确率不变时提高正确答案置信度，损失仍下降。这里解释的是题目中的概率变化，不把高置信度一概等同于可靠校准。

复查脚本与参考逻辑一致，只清理末尾空白并通过语法树相等检查，不改计算行为。所有非空代码行保留中文注释。CI 新增 scripts/ci/check_lesson35.py：只读调用 main 重新统计训练目标、复算完整验证损失和命中数，在浮点容差内比较困惑度与准确率，并核对报告内公式关系及原模型文件和报告未变。完成后按约定完整上传代码、正式报告、笔记与检查配置，本课不新增权重文件。

## 第三十六课：学习率的受控对照实验

学习者尚未完成本课。新建 scripts/learn/20_learning_rate_experiment.py，由学习者按四块手输；老师预跑只写临时目录，不替代实操，不创建正式报告。两组从 lesson33_best.npz 同一权重出发，固定同一章节打乱种子45、同一前200批、批大小16、窗口32、SGD和裁剪阈值1，仅学习率分别0.1和0.03。每组新增200步，累计步数为14561；不是先训练0.1再接着训练0.03，也不是完整第4轮。

更新公式 θ_new=θ_old−ηg_clipped。在相同起点和第一批上，梯度及裁剪一致，学习率0.03的更新长度是0.1的0.3倍；后续参数已经分叉，梯度也可能不同，不能认为所有后续更新向量都成固定比例。较小学习率不保证更好，也不直接减少每步计算量；固定200步比较的是给定预算的短期效果，不是最终收敛上限。

先固定候选和步数再看结果。正常多轮连续训练只在外面加载模型；这里外层循环代表两组独立实验，因此每组都要重新加载共同起点。组内批次循环仍保持同一个模型持续更新。原模型本身也作为候选，若两组验证损失都更差就保留原模型。每组只观察最终检查点，没有声称组内每一步的最优值。使用验证集做模型选择是允许的，但多次选择会间接利用验证信息；测试集继续留作后续最终评估。

缓存200批 X/Y 的数组大小为200×16×32×2×8=1638400字节，约1.56MiB；这个数字不包含字典、章节标识、模型、梯度和前向中间数组等内存。缓存是为了精确复用相同训练批次，数组设为只读以防改动。实际书籍覆盖需要报告，不把随机章节前缀当成四书均衡采样。

### 第一块：固定实验配置

```python
from importlib import import_module  # 复用已有训练、验证和生成函数。
from itertools import islice  # 只取预先规定的两百个训练批次。
from pathlib import Path  # 定位数据、权重及报告。
import hashlib  # 检查起点权重文件没有被覆盖。
import json  # 读取已有配置并保存对照实验。
import random  # 固定本次训练章节的打乱顺序。
import numpy as np  # 核对参数并检查损失是否有效。

ROOT = Path(__file__).resolve().parents[2]  # 从 scripts/learn 定位项目根目录。
network = import_module("07_tiny_rnn")  # 使用相同模型及梯度裁剪。
training = import_module("11_train_steps")  # 复用加载、保存和固定参数评估。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证评估函数。
decoding = import_module("12_generate_text")  # 保持相同生成设置观察文字。
STEPS = 200  # 每组只更新两百步，不把这次短实验称为完整一轮。
RATES = (("lr010", 0.1), ("lr003", 0.03))  # 在看结果之前固定两种学习率。
```

### 第二块：复用同一个分支的多步更新

```python
def run_updates(model, batches, learning_rate):  # 在指定批次上连续更新同一个模型。
    for local_step, batch in enumerate(batches, start=1):  # 两组都从相同列表的第一批开始。
        loss, raw_grads = model.loss_and_grad(batch["x"], batch["y"])  # 用当前参数计算损失和完整梯度。
        assert np.isfinite(loss), "训练损失无效"  # 非有限损失不能继续更新。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 两组都使用同一全局裁剪阈值。
        for name in model.params:  # 完整反向结束后再更新五组参数。
            model.params[name] -= learning_rate * grads[name]  # 本实验唯一主动改变的训练设置是学习率。
        assert all(np.isfinite(value).all() for value in model.params.values()), "更新后参数无效"  # 及时发现数值异常。
        if local_step % 50 == 0:  # 每五十步显示一次进度。
            print(f"lr={learning_rate} step={local_step}/{STEPS} batch_loss={loss:.6f}", flush=True)  # 当前批次损失不能替代完整验证损失。
```

### 第三块：相同起点、相同数据、独立训练和比较

```python
def main(output_dir=None):  # 从共同起点比较两条独立训练分支。
    data = import_module("06_make_windows")  # 复用训练词表、窗口与批次函数。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 使用已经选出的三轮最佳模型。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 保证实验来源明确。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 旧模型只读，新增分支使用不同文件名。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 记录源文件完整哈希。
    initial, vocab, start_step = training.load_checkpoint(source_path, network)  # 加载共同起点用于核对。
    assert vocab == data.vocab, "模型词表不一致"  # 输入编号必须对应正确表示表。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 默认保存到 model，也允许老师预跑时使用临时目录。
    records = list(data.train_records)  # 从同一原始章节顺序复制。
    random.Random(45).shuffle(records)  # 两组使用同一预先固定的章节顺序。
    batches = list(islice(data.iter_batches(data.iter_windows(records, 32, 32), 16), STEPS))  # 缓存相同的两百批，避免两组使用不同数据。
    assert len(batches) == STEPS and sum(batch["y"].size for batch in batches) == 102400, "训练规模不正确"  # 每组两百批共十万两千四百个目标位置。
    for batch in batches:  # 将缓存的输入和目标都设为只读。
        batch["x"].setflags(write=False)  # 防止训练过程意外修改共享输入。
        batch["y"].setflags(write=False)  # 防止训练过程意外修改共享正确答案。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 准备统计这段训练实际覆盖哪些书。
    book_windows = {record["book"]: 0 for record in records}  # 没有覆盖到的书也保留零计数。
    for batch in batches:  # 遍历缓存的训练来源信息。
        for chapter_id in batch["chapter_ids"]:  # 每一行对应一个训练窗口。
            book_windows[chapter_to_book[chapter_id]] += 1  # 短实验不假定四本书覆盖均衡。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 所有候选都使用同一完整验证集。
    initial_loss, positions = epoch_tools.evaluate_full(initial, data, validation)  # 重新测量共同起点，防止继续训练反而变差却被忽略。
    assert positions == 316960 and np.isclose(initial_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点指标不一致"  # 核对完整验证范围及权重。
    history = [{"label": "starting_model", "checkpoint": source_path.name, "completed_steps": start_step, "validation_loss": initial_loss}]  # 原始模型也参与最佳选择。
    print("共同起点完整验证损失：", initial_loss, flush=True)  # 两组使用同一条起跑线。
    for label, learning_rate in RATES:  # 每次循环是一组独立实验，不是同一模型的连续阶段。
        model, current_vocab, loaded_step = training.load_checkpoint(source_path, network)  # 每组开始重新加载共同起点。
        assert current_vocab == vocab and loaded_step == start_step and all(np.array_equal(value, initial.params[name]) for name, value in model.params.items()), "分支起点不同"  # 不能让第二组接着第一组训练。
        run_updates(model, batches, learning_rate)  # 只在本组内部连续更新。
        loss, current_positions = epoch_tools.evaluate_full(model, data, validation)  # 使用完整验证损失评价本组最终参数。
        assert current_positions == positions, "验证范围不一致"  # 同一分母才能公平比较。
        path = output_dir / f"lesson36_{label}.npz"  # 为两组分别命名，保留原权重。
        training.save_checkpoint(path, model, vocab, start_step + STEPS)  # 每组都只在共同起点上新增两百步。
        greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的开头和生成长度。
        history.append({"label": label, "learning_rate": learning_rate, "checkpoint": path.name, "completed_steps": start_step + STEPS, "validation_loss": loss, "change_from_start": loss - initial_loss, "greedy": greedy})  # 同时保存分数、变化量和生成观察。
        print(f"lr={learning_rate} full_validation={loss:.6f} change={loss-initial_loss:+.6f}", flush=True)  # 负变化量表示优于起点。
        print("贪心：", repr(greedy), flush=True)  # 文字不通顺或重复也如实保留。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "共同起点文件被修改"  # 不允许新实验覆盖旧最佳权重。
    assert all(np.array_equal(value, initial.params[name]) for name, value in training.load_checkpoint(source_path, network)[0].params.items()), "共同起点参数被改变"  # 保留的起点对象也不能被分支更新影响。
    selected = min(history, key=lambda row: row["validation_loss"])  # 在起点和两组最终检查点中选最低分，相等保留较早候选。
    report = {  # 记录固定预算的短实验，而不是宣称找到全局最佳学习率。
        "starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"],  # 保存模型与数据来源。
        "config": {"steps_per_branch": STEPS, "learning_rates": [rate for _, rate in RATES], "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45},  # 其余训练条件固定。
        "training_positions_per_branch": 102400, "cached_xy_bytes": sum(batch["x"].nbytes + batch["y"].nbytes for batch in batches), "book_windows": book_windows,  # 记录数据规模、数组内存及实际书籍覆盖。
        "validation_positions": positions, "history": history, "selected_label": selected["label"], "selected_checkpoint": selected["checkpoint"],  # 被选中的也可能仍是原模型。
    }  # 不使用测试集，不修改上一课的最佳模型记录。
    print("本次候选中选中：", selected["label"], selected["validation_loss"], flush=True)  # 只在本实验观察过的三个候选之间选择。
    print("实际训练窗口覆盖：", book_windows, flush=True)  # 明确短训练不等同于完整遍历。
    return report  # 返回结果供入口保存。
```

### 第四块：保存正式实验报告

```python
if __name__ == "__main__":  # 直接运行时才训练本课两个分支并保存报告。
    result = main()  # 默认把分支权重写入 model 目录。
    path = ROOT / "docs/training_runs/lesson36.json"  # 使用独立实验报告，不覆盖旧记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存可读 JSON。
    print("实验报告：", path)  # 输出结果位置。
```

### 实操与问题

运行 python scripts/learn/20_learning_rate_experiment.py。将生成 model/lesson36_lr010.npz、model/lesson36_lr003.npz 和 docs/training_runs/lesson36.json。重新运行从同一旧起点重做两组并覆盖本课产物，不接续上次分支。请提交两组完整验证损失、相对起点变化及最后选中候选。问题一：第二组应该从第一组训练结束后的参数开始吗？问题二：若起点6.0、学习率0.1分支6.1、学习率0.03分支6.05，应该保留哪个候选？检查后再添加明确的权重忽略例外并完整上传。

### 老师完整预跑

两个训练分支及完整验证均已运行完成，源文件哈希与起点参数保留检查通过。每组训练102400个目标、3200个窗口；实际窗口覆盖：西游记1014、红楼梦1532、三国演义202、水浒全传452。这是同一段章节前缀，覆盖不均衡，应限制结论范围。预跑使用临时目录，完成后已删除，不留下学习者正式权重或报告。

| 候选 | 完整验证损失（保留六位小数） | 相对起点变化 |
|---|---:|---:|
| 起点 | 6.011733 | 0 |
| 学习率0.1 | 5.984847 | -0.026886 |
| 学习率0.03 | 5.960013 | -0.051719 |

本次选中lr003，未舍入的该分支损失为5.9600134563109055。两组贪心仍然循环重复：“只是那个，不是”和“不是我，一个”等。因此验证改善不等同于生成质量达标；不能推出0.03在所有任务或更长训练中最优。

训练与验证已经完整通过；预跑包装脚本最后补写笔记时缺少json导入，仅影响笔记追加，已根据实际终端输出补齐这里的记录，不重复训练来修复文档。参考练习代码本身有json导入且报告入口已通过语法检查。

### 学习者第三十六课实操验收与选择规则纠正

学习者完成两组实验，脚本语法树与已预跑参考代码一致，只有空白差异；清理末尾空白不改变计算逻辑，所有非空代码行保留直白注释。正式报告 lesson36.json：共同起点6.011732826733577，0.1分支5.9848467574584125（变化-0.02688606927516446），0.03分支5.9600134563109055（变化-0.051719370422671496），实际选中lr003正确。每组从相同起点新增200步，累计14561步；相同训练前缀为102400个预测位置，缓存X/Y字节数1638400，四书窗口计数1014/1532/202/452与预跑一致。

第1题回答“不应该”正确：第二组不能接着第一组训练，必须重新加载共同起点。第2题回答“6.05”不正确：题设起点6.0、新分支6.1和6.05，最小值仍是起点6.0，应保留旧模型。6.05只是在两个新分支之间更好，但仍比旧模型差。选优必须覆盖全部候选，包括未继续训练的起点；继续训练不保证验证改善。

本课实际两个分支都优于起点，所以选lr003；这与题设答案要保留起点并不矛盾。本次知识确认停留在这一选择规则，等待学习者进一步确认，不提前记为已掌握。

复查采用只读检查 scripts/ci/check_lesson36.py：独立重建200批训练前缀与书籍覆盖，校验共同起点SHA-256和分支累计步数，复算起点及两个最终检查点的完整验证损失，检查变化量和包括起点的最低损失选择。生成流程核对合法编号与长度，不比较跨平台整段字符串，不重新训练400步。代码、报告、两份权重、模型说明、笔记和CI配置按每课完整上传约定提交。

### 第三十六课选择规则确认

对于旧模型5.8、两个新分支5.9和5.85的例子，学习者回答“保留旧模型损失5.8”正确。这一例子中已能把旧模型纳入全部候选选择。

## 第三十七课：学习率衰减后的完整第四轮训练

学习者尚未实操。新建 scripts/learn/21_train_with_decay.py，由学习者手输。前三轮0.1，第四轮使用0.03，这是一种按训练阶段降低学习率的衰减安排。这里不是指数衰减或每步下降，本轮内部学习率固定。第36课只比较200步，不能保证更长训练仍然改善，本课检验整轮轨迹。没有额外训练完整0.1第四轮对照，因此不据此证明整轮0.03一定优于0.1。

重新从lesson33_best.npz累计14361步出发，按种子45顺序完成4787次更新，最终累计19148步。不能从lesson36_lr003.npz出发又从第四轮第一批重新喂全部数据，那会重复训练前200批。这里从原三轮模型重做前200步，主动核对其验证损失与上一课lr003一致，再顺着同一生成器完成剩余4587步。

预先指定局部步数0、200、1000、3000、4787为候选，全部用完整316960个验证位置。局部步数是本轮更新次数，累计步数=14361+局部步数。起点也参加选优，严格改善才更新独立lesson37_best.npz。最后重载最佳文件逐参数对照所选候选。这里没有自动早停，也不声称找到未评估步骤中的最优值。

每轮窗口76584、目标2450688、尾批8行，模型结构和719723个参数不变。降低学习率不改变每步矩阵计算规模；额外完整验证会增加运行时间。run_seconds包括起点验证、训练、周期验证、生成、保存和核对，不能当作纯训练吞吐耗时。报告在检查点后落盘，但重跑默认仍从第33课起点开始，不自动从中断位置恢复。

### 第一块：导入和学习率

```python
from importlib import import_module  # 复用模型、权重、验证和生成工具。
from pathlib import Path  # 定位项目文件与实验输出。
from time import perf_counter  # 记录含验证与保存的整次运行时间。
import hashlib  # 检查旧最佳文件未被覆盖。
import json  # 读取来源报告并保存新记录。
import random  # 固定第四轮的章节顺序。
import numpy as np  # 检查数值及重新加载后的参数。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn。
network = import_module("07_tiny_rnn")  # 继续使用同一个模型结构。
training = import_module("11_train_steps")  # 复用权重保存与加载。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证与报告保存。
decoding = import_module("12_generate_text")  # 固定生成条件比较文字。
LEARNING_RATE = 0.03  # 前三轮使用0.1，本轮降低为0.03。
```

### 第二块：完整验证检查点

```python
def record_checkpoint(model, vocab, data, validation, local_step, global_step, output_dir):  # 测量并保存指定训练位置。
    loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 每次都评估相同完整验证窗口集。
    assert positions == 316960, "验证范围变化"  # 不混用子集指标。
    path = output_dir / f"lesson37_step{local_step:04d}.npz"  # 文件名记录本轮局部步数。
    training.save_checkpoint(path, model, vocab, global_step)  # 文件元数据记录累计步数。
    greedy, _ = decoding.generate(model, vocab, "春风", new_tokens=40, context_size=32)  # 保持原来的贪心设置。
    print(f"检查点 local_step={local_step} global_step={global_step} full_validation={loss:.6f}", flush=True)  # 立即显示长训练中的验证结果。
    print("贪心：", repr(greedy), flush=True)  # 如实保留重复和不通顺。
    return {"local_step": local_step, "global_step": global_step, "checkpoint": path.name, "validation_loss": loss, "validation_positions": positions, "greedy": greedy}  # 只保存可序列化的记录。
```

### 第三块：准备起点和整轮数据

```python
def main(output_dir=None, report_path=None):  # 默认保存正式产物，老师预跑时可指定临时目录。
    started = perf_counter()  # 运行耗时包括验证、生成及保存。
    data = import_module("06_make_windows")  # 复用固定词表与训练窗口。
    source = json.loads((ROOT / "docs/training_runs/lesson33.json").read_text(encoding="utf-8"))  # 从完整三轮的最佳模型出发。
    short_run = json.loads((ROOT / "docs/training_runs/lesson36.json").read_text(encoding="utf-8"))  # 读取上一课短实验用于核对前两百步。
    assert source["status"] == "completed" and source["best_epoch"] == 3 and source["train_sha256"] == short_run["train_sha256"] == data.vocab["train_sha256"], "起点或数据版本不正确"  # 本课明确从三轮完整模型开始。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 不直接从上一课的两百步分支开始。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 保留旧文件校验值。
    model, vocab, start_step = training.load_checkpoint(source_path, network)  # 训练模型只在循环之前加载一次。
    assert vocab == data.vocab and start_step == 14361, "起点权重配置不一致"  # 起点已经完成三轮。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 正式权重默认放入model。
    report_path = ROOT / "docs/training_runs/lesson37.json" if report_path is None else Path(report_path)  # 正式报告与旧实验分开。
    best_path = output_dir / "lesson37_best.npz"  # 最佳文件是独立快照。
    records = list(data.train_records)  # 从原始章节列表复制。
    random.Random(45).shuffle(records)  # 与上一课短实验相同的第四轮顺序。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 全程使用相同完整验证集。
    windows_per_book = {}  # 独立计算本轮应覆盖的窗口数量。
    for record in records:  # 按章节长度计算，避免依赖已训练计数。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 每个窗口需要三十三字才能产生三十二个目标。
        windows_per_book[record["book"]] = windows_per_book.get(record["book"], 0) + count  # 汇总每本书的窗口。
    expected_windows = sum(windows_per_book.values())  # 得到整轮窗口数。
    expected_steps = (expected_windows + 15) // 16  # 保留尾批，批次数向上取整。
    initial_loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 起点也必须参加选优。
    assert positions == 316960 and np.isclose(initial_loss, source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点验证不一致"  # 核对加载和评估范围。
    history = [{"local_step": 0, "global_step": start_step, "checkpoint": source_path.name, "validation_loss": initial_loss, "validation_positions": positions}]  # 零步表示本轮尚未更新。
    best_row = history[0]  # 如果后续全部变差，仍保留起点。
    training.save_checkpoint(best_path, model, vocab, start_step)  # 初始化独立最佳文件。
    report = {"status": "running", "starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"], "config": {"epoch": 4, "learning_rate": LEARNING_RATE, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 45}, "history": history, "completed_local_steps": 0, "best_local_step": 0, "best_checkpoint": best_path.name}  # 新报告保留配置和全部已评估候选。
    epoch_tools.save_report(report_path, report)  # 先保存起点状态。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 用于累计实际训练来源。
    book_windows = {book: 0 for book in windows_per_book}  # 实际覆盖从零开始。
    seen_windows, seen_positions, loss_sum = 0, 0, 0.0  # 窗口、目标及训练过程损失分别统计。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 本轮完整遍历，不只取前两百批。
```

### 第四块：训练与定期选择

```python
    for local_step, batch in enumerate(batches, start=1):  # 每个批次更新一次，局部步数从一开始。
        x, y = batch["x"], batch["y"]  # 本批实际输入和目标。
        loss, raw_grads = model.loss_and_grad(x, y)  # 在当前参数上完成前向与反向。
        assert np.isfinite(loss), "训练损失无效"  # 无效数值不能继续传播。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 与之前相同的全局裁剪。
        for name in model.params:  # 完整反向结束后更新五组参数。
            model.params[name] -= LEARNING_RATE * grads[name]  # 本轮所有步骤都使用0.03。
        assert all(np.isfinite(value).all() for value in model.params.values()), "参数无效"  # 及时发现异常更新。
        seen_windows += x.shape[0]  # 按实际批大小累计窗口数。
        seen_positions += y.size  # 尾批只计实际目标。
        loss_sum += loss * y.size  # 训练过程损失按实际位置加权。
        for chapter_id in batch["chapter_ids"]:  # 各行归属对应书籍。
            book_windows[chapter_to_book[chapter_id]] += 1  # 检查完整四书覆盖。
        if local_step % 500 == 0:  # 每五百步显示训练进度。
            print(f"local_step={local_step}/{expected_steps} global_step={start_step+local_step} batch_loss={loss:.6f}", flush=True)  # 不要求不同批次损失单调下降。
        if local_step in (200, 1000, 3000, expected_steps):  # 只在预先规定的位置评估完整验证集。
            row = record_checkpoint(model, vocab, data, validation, local_step, start_step + local_step, output_dir)  # 当前参数固定后评估保存。
            if local_step == 200:  # 相同起点、数据顺序及学习率应复现上一课前缀指标。
                previous = next(item for item in short_run["history"] if item["label"] == "lr003")  # 取上一课0.03分支。
                assert np.isclose(row["validation_loss"], previous["validation_loss"], rtol=1e-6, atol=1e-6), "前两百步与短实验不一致"  # 评估不更新参数，插入检查不应改变训练轨迹。
            history.append(row)  # 所有预定候选都保留。
            if row["validation_loss"] < best_row["validation_loss"]:  # 只在严格改善时替换最佳文件。
                best_row = row  # 保留对应局部和累计步数。
                training.save_checkpoint(best_path, model, vocab, row["global_step"])  # 独立快照不会随后续训练变化。
            report.update({"completed_local_steps": local_step, "best_local_step": best_row["local_step"], "best_validation_loss": best_row["validation_loss"]})  # 同步当前完整检查点进度。
            epoch_tools.save_report(report_path, report)  # 每个检查点后保存报告。
```

### 第五块：完整覆盖、最佳文件重载和入口

```python
    assert local_step == expected_steps == 4787 and seen_windows == expected_windows == 76584, "整轮覆盖不正确"  # 本轮必须完整训练。
    assert seen_positions == 2450688 and x.shape[0] == 8 and book_windows == windows_per_book, "目标、尾批或书籍覆盖错误"  # 不能漏掉尾批或某一本书。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "旧起点被覆盖"  # 新实验保留旧文件。
    selected = min(history, key=lambda item: item["validation_loss"])  # 包括起点在内独立核对最低分。
    assert selected["local_step"] == best_row["local_step"], "最佳选择不正确"  # 最后一步未必是最佳一步。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后重新加载实际最佳文件。
    selected_path = source_path if selected["local_step"] == 0 else output_dir / selected["checkpoint"]  # 最佳也可能仍是旧起点。
    selected_model, _, selected_step = training.load_checkpoint(selected_path, network)  # 加载候选文件作参数对照。
    assert best_vocab == vocab and best_step == selected_step == selected["global_step"] and all(np.array_equal(value, selected_model.params[name]) for name, value in best_model.params.items()), "最佳文件与所选候选不一致"  # 防止最佳报告和实际文件不匹配。
    report.update({"status": "completed", "completed_global_steps": start_step + local_step, "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": x.shape[0], "book_windows": book_windows, "training_process_loss": loss_sum / seen_positions, "run_seconds": perf_counter() - started})  # 记录完整覆盖及含验证保存的运行时间。
    epoch_tools.save_report(report_path, report)  # 全部核对完成后标记实验结束。
    print("本轮及累计更新步数：", local_step, report["completed_global_steps"], flush=True)  # 预期为4787和19148。
    print("最佳局部步数及完整验证损失：", best_row["local_step"], best_row["validation_loss"], flush=True)  # 仅在本课已评估候选中选择。
    print("第四轮覆盖、短实验前缀与最佳快照检查：通过", flush=True)  # 不将验证改善等同于流畅生成。
    return report  # 返回结果供检查工具使用。


if __name__ == "__main__":  # 直接运行才训练和保存正式产物。
    main()  # 默认写入model与docs/training_runs。
```

### 实操任务

运行python scripts/learn/21_train_with_decay.py。将生成lesson37_step0200.npz、lesson37_step1000.npz、lesson37_step3000.npz、lesson37_step4787.npz和lesson37_best.npz，以及docs/training_runs/lesson37.json。老师完整预跑只写临时目录，不创建学习者练习或正式结果。请提交各检查点完整验证损失、最后累计步数与最佳局部步数。问题一：本轮局部第1000步对应累计多少步？问题二：如果局部3000步验证损失比4787步更低，最佳应该选哪个，而完整一轮是否仍已完成？学习者完成后再检查、添加权重例外并完整上传。

### 老师完整预跑

完整预跑通过，写入的临时权重和报告均已清理，未替学习者创建正式脚本。检查点结果如下：

| 本轮局部步数 | 累计步数 | 完整验证损失 |
|---:|---:|---:|
| 0 | 14361 | 6.011732827 |
| 200 | 14561 | 5.960013456 |
| 1000 | 15361 | 5.971619851 |
| 3000 | 17361 | 5.939247507 |
| 4787 | 19148 | 5.937874744 |

最佳局部步数4787，最佳完整验证损失5.937874743698944。完整覆盖76584窗口、2450688目标，尾批8行；含验证和保存总耗时约625.7秒。新增权重大小记录：{"lesson37_best.npz": 2687778, "lesson37_step0200.npz": 2687807, "lesson37_step1000.npz": 2687770, "lesson37_step3000.npz": 2687815, "lesson37_step4787.npz": 2687778}字节。

完整临时报告参考：

```json
{
  "status": "completed",
  "starting_checkpoint": "lesson33_best.npz",
  "starting_sha256": "3eebe534171dbb7c660052763d00c4ecff705d2609da3dd33403a90fc9b8a919",
  "train_sha256": "0cee762b4832c3a1c37c84c245ca912c51a1c22fa8df4c37e811ec735e390a69",
  "config": {
    "epoch": 4,
    "learning_rate": 0.03,
    "batch_size": 16,
    "seq_length": 32,
    "max_norm": 1.0,
    "chapter_shuffle_seed": 45
  },
  "history": [
    {
      "local_step": 0,
      "global_step": 14361,
      "checkpoint": "lesson33_best.npz",
      "validation_loss": 6.011732826733577,
      "validation_positions": 316960
    },
    {
      "local_step": 200,
      "global_step": 14561,
      "checkpoint": "lesson37_step0200.npz",
      "validation_loss": 5.9600134563109055,
      "validation_positions": 316960,
      "greedy": "春风，只是这个，不是我，一个，不是我，一个，不是我，一个，不是我，一个，不是我，一个"
    },
    {
      "local_step": 1000,
      "global_step": 15361,
      "checkpoint": "lesson37_step1000.npz",
      "validation_loss": 5.97161985105604,
      "validation_positions": 316960,
      "greedy": "春风，只不是道：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道：“我不道"
    },
    {
      "local_step": 3000,
      "global_step": 17361,
      "checkpoint": "lesson37_step3000.npz",
      "validation_loss": 5.939247507329302,
      "validation_positions": 316960,
      "greedy": "春风，只是一个，不是一个，不是一个，不是一个，不是一个，不是一个，不是一个，不是一个"
    },
    {
      "local_step": 4787,
      "global_step": 19148,
      "checkpoint": "lesson37_step4787.npz",
      "validation_loss": 5.937874743698944,
      "validation_positions": 316960,
      "greedy": "春风，只是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人"
    }
  ],
  "completed_local_steps": 4787,
  "best_local_step": 4787,
  "best_checkpoint": "lesson37_best.npz",
  "best_validation_loss": 5.937874743698944,
  "completed_global_steps": 19148,
  "seen_windows": 76584,
  "seen_positions": 2450688,
  "last_batch_size": 8,
  "book_windows": {
    "西游记.txt": 18052,
    "红楼梦.txt": 21299,
    "三国演义.txt": 14818,
    "水浒全传.txt": 22415
  },
  "training_process_loss": 5.878568746721584,
  "run_seconds": 625.6712169999955
}
```

### 学习者第三十七课实操验收与步数讲解

学习者已完成第四轮，报告lesson37.json状态completed。起点完整验证损失6.011732826733577，局部200步5.9600134563109055、1000步5.97161985105604、3000步5.939247507329302、4787步5.937874743698944；本次选4787步正确。累计19148步，完整覆盖76584窗口、2450688目标、尾批8行，四书实际数量与独立计算一致。训练过程按位置加权损失5.878568746721584；实际运行393.7秒包含验证和保存，不能当作纯训练吞吐耗时。

学习者两题回答“我不知道”，因此运行完成与概念掌握分开记录。第1题：局部第1000步=在本轮又更新1000次；起点已更新14361次，所以累计14361+1000=15361。日志同时显示local_step=1000和global_step=15361，表示同一次训练更新位置的两种计数。

第2题：假设局部3000步验证更低，就选3000步快照；只要实际训练已跑完4787步，整轮仍已完成。已完成多少训练描述执行过程，最终选哪个文件描述模型选择，两个问题互不冲突。好比完成全部练习后，保留中途表现最好的那份答案；选择旧快照不会抹去之后已经执行的更新。最佳文件元数据记录所选位置的累计步数，而报告completed_global_steps记录这次实际训练完成的累计步数，两者可以不同。

概念确认只保留一道具体小题：起点14361步，本轮又训练200步，累计多少步？尚未收到回答，不提前记为理解。

脚本语法树与已完整预跑参考代码一致，仅清理末尾空白，所有非空代码行仍有中文注释。只读CI新增check_lesson37.py：核对完整覆盖、五个候选的完整验证损失、局部200步与第36课低学习率分支参数一致、最佳文件与所选候选完全相同，不重训第四轮。按每课完整上传约定提交代码、报告、五份权重、说明和检查配置。

### 第三十七课局部与累计步数确认

学习者对“起点14361次，本轮再更新200次”的问题回答“14561次”正确。局部200与累计14561的这个例子已答对，不据此提前记录已理解全部检查点选择规则。

## 第三十八课：早停、等待次数与最小改善量

学习者尚未实操。本课新建scripts/learn/22_early_stopping_demo.py，只回放第37课的已保存完整验证记录，输出docs/training_runs/lesson38.json。不训练、不改权重，也没有真的节省已经执行的训练计算。模拟中每次只能访问当前及过去记录，达到停止规则即break，不再把后续记录用于该组选择。真实训练中早停应放在验证后、下一次参数更新前判断。

patience表示允许连续多少次验证检查未达到明显改善条件，达到该次数触发停止。它的单位是验证检查次数，不是训练步数，也不自动表示轮数。本课检查位置0/200/1000/3000/4787并不等间隔，所以patience=2不代表固定多少更新预算。起点只初始化，不算一次失败。

min_delta使用交叉熵的绝对改善量，本课显著改善定义为reference_loss−current_loss严格大于min_delta，相等不计为明显改善。“明显”仅指达到该预设阈值，不是统计显著性。有效改善时参考值更新并将bad_checks清零，否则bad_checks加一。参考值只在有效改善时更新，因此数次小改善可累计到超过参考值的阈值。

两种职责分开：best记录已经看到的最低原始验证损失，任何严格更低都更新；reference_loss用于判断是否有足够进展以继续等待。比如5.939248到5.937875仅改善约0.001373，小于0.005，此时仍保存更低的最佳模型，但不清零未明显改善次数。某些框架会用同一min_delta决定最佳权重更新，本课明确采用分开管理的定义。

早停不保证找到整个未来训练轨迹的最低点。第37课在1000步短暂变差、3000步又改善，patience=1会在1000步触发停止并恢复200步模型，从而错过后续更低分；真实训练无法预知未来，所以这是节省计算与容忍短暂波动之间的权衡。早停策略是事前约定的训练规则，本课三组仅作教学比较，不因已知未来结果宣称patience=2是普遍最优。继续留出测试集，不用测试分数决定何时停止。

### 第一块：初始化早停状态

```python
from pathlib import Path  # 从当前脚本位置找到项目根目录。
import json  # 读取旧验证记录，并保存本课模拟结果。
import math  # 检查损失和阈值是否为有限数。

ROOT = Path(__file__).resolve().parents[2]  # 本文件放在 scripts/learn。


def simulate_early_stopping(history, patience, min_delta):  # 按时间顺序模拟早停，不训练模型。
    assert isinstance(patience, int) and patience >= 1, "耐心次数至少为一"  # patience按验证检查次数计数。
    assert math.isfinite(min_delta) and min_delta >= 0, "改善阈值不能为负"  # min_delta使用交叉熵的绝对差值。
    assert len(history) >= 1, "没有验证记录"  # 至少需要一条起点记录。
    best = dict(history[0])  # 起点也是已经观察过的最佳候选。
    reference_loss = best["validation_loss"]  # 早停参考值只在改善足够明显时更新。
    bad_checks = 0  # 连续没有明显改善的检查次数。
    actions = []  # 记录每次实际访问的验证检查和决定。
```

### 第二块：逐次判断和返回结果

```python
    for row in history[1:]:  # 从起点之后的第一条验证记录开始逐条观察。
        loss = row["validation_loss"]  # 此时只能看到当前分数，不能提前使用未来记录。
        new_best = loss < best["validation_loss"]  # 任何严格更低的损失都值得保留快照。
        if new_best:  # 更新用于最终模型选择的最低分记录。
            best = dict(row)  # 复制普通记录，不修改原始历史。
        significant = reference_loss - loss > min_delta  # 本课规定改善量必须严格大于阈值。
        if significant:  # 相比上次有效参考值已有足够改善。
            reference_loss = loss  # 更新早停参考值。
            bad_checks = 0  # 有明显改善，连续未改善次数清零。
        else:  # 变差、持平或改善不足阈值都计入等待。
            bad_checks += 1  # 不是训练步数加一，而是验证检查次数加一。
        should_stop = bad_checks >= patience  # 达到预先规定的等待次数就触发停止。
        actions.append({"local_step": row["local_step"], "validation_loss": loss, "new_best": new_best, "significant": significant, "reference_loss": reference_loss, "bad_checks": bad_checks, "should_stop": should_stop})  # 记录判断依据。
        if should_stop:  # 真实训练将在这里跳出训练过程。
            break  # 本模拟也不再读取后面的验证分数。
    last_seen = history[len(actions)]  # 起点不计入actions，因此此下标对应最后观察的位置。
    return {  # 记录停止位置与选中模型，二者可以不同。
        "patience": patience, "min_delta": min_delta,  # 保留本组规则。
        "stop_triggered": bool(actions and actions[-1]["should_stop"]),  # 是否触发停止，而非宣称实际撤销了训练。
        "last_seen_local_step": last_seen["local_step"], "best_local_step": best["local_step"],  # 区分停止位置和最低分位置。
        "best_checkpoint": best["checkpoint"], "best_validation_loss": best["validation_loss"],  # 模拟中应恢复的最佳候选。
        "reference_loss": reference_loss, "actions": actions,  # 同时保留早停参考值和逐次决定。
    }  # 模型选择的最低分和早停的改善阈值分别管理。
```

### 第三块：读取第四轮记录并回放

```python
def main():  # 使用已经完成的第四轮报告回放三组早停规则。
    source_path = ROOT / "docs/training_runs/lesson37.json"  # 只读取已有验证记录。
    source_bytes = source_path.read_bytes()  # 保留原报告以检查文件未变。
    source = json.loads(source_bytes)  # 将JSON还原成字典。
    assert source["status"] == "completed", "上一课尚未完成"  # 本课分析已完成的轨迹。
    history = source["history"]  # 沿用原来的验证时间顺序，不按损失排序。
    assert history and history[0]["local_step"] == 0, "缺少起点"  # 第一条记录用于初始化参考值。
    assert all(math.isfinite(row["validation_loss"]) and row["validation_loss"] >= 0 and row["validation_positions"] == 316960 for row in history), "验证记录无效"  # 所有分数来自同一完整验证范围。
    assert all(left["local_step"] < right["local_step"] for left, right in zip(history, history[1:])), "记录不是时间顺序"  # 防止把未来最低分提前放到前面。
    results = []  # 保存三种教学规则的模拟结果。
    for patience, min_delta in ((1, 0.0), (2, 0.0), (2, 0.005)):  # 用不同规则理解等待次数和改善阈值，不据此宣称找到最佳策略。
        result = simulate_early_stopping(history, patience, min_delta)  # 每组都从起点重新模拟。
        results.append(result)  # 保留该组完整判断过程。
        print(f"patience={patience} min_delta={min_delta}", flush=True)  # 显示本组规则。
        for action in result["actions"]:  # 只打印这组实际观察过的记录。
            print(f"  local_step={action['local_step']} loss={action['validation_loss']:.6f} new_best={action['new_best']} significant={action['significant']} bad_checks={action['bad_checks']} stop={action['should_stop']}", flush=True)  # 展示一次次决定，而非只看结论。
        print("触发停止：", result["stop_triggered"], "最后观察位置：", result["last_seen_local_step"], "应加载最佳位置：", result["best_local_step"], flush=True)  # 停止时不一定使用最后观察到的模型。
    assert source_path.read_bytes() == source_bytes, "旧报告被修改"  # 模拟不能覆盖真实训练记录。
    return {"source_report": source_path.name, "train_sha256": source["train_sha256"], "mode": "replay_without_training", "results": results}  # 明确这只是回放，没有实际节省已发生的训练。
```

### 第四块：保存模拟报告

```python
if __name__ == "__main__":  # 直接运行时才模拟并保存本课结果。
    report = main()  # 不加载模型、不更新参数、不写权重。
    path = ROOT / "docs/training_runs/lesson38.json"  # 保存独立的教学模拟报告。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存中文可读JSON。
    print("早停模拟报告：", path)  # 输出本课报告路径。
```

### 实操与问题

第一块顶层导入及函数定义；第二块继续simulate_early_stopping函数，保持四格缩进；第三块main回到顶层；第四块入口同样在顶层。运行python scripts/learn/22_early_stopping_demo.py，一般很快完成，不需要再次等待完整训练。请贴三组的“触发停止、最后观察位置、应加载最佳位置”。问题一：patience=2时仅连续一次没有明显改善，应该立即停止吗？问题二：若在1000步触发停止，但观察到的最低损失属于200步，应加载哪个位置的文件？完成后检查、记录并完整上传脚本及报告。

### 老师预跑结果

参考代码已在内存执行，没有创建学习者练习脚本或正式报告。

| patience | min_delta | 是否触发停止 | 最后观察局部步数 | 应恢复最佳局部步数 |
|---:|---:|---|---:|---:|
| 1 | 0.0 | True | 1000 | 200 |
| 2 | 0.0 | False | 4787 | 4787 |
| 2 | 0.005 | False | 4787 | 4787 |

第三组在最后一步new_best=True、significant=False、bad_checks=1；早停参考值仍是3000步损失，但最低分文件属于4787步。停止位置和恢复位置、参考值和原始最低分的分离检查均通过，旧第37课报告字节未变。

### 学习者第三十八课实操验收与回答

学习者提交完整输出与lesson38.json，三组回放与老师预跑一致：patience=1/min_delta=0在1000步触发停止、应选200步；patience=2/min_delta=0与patience=2/min_delta=0.005均未触发停止、最终选4787步。第三组末次new_best=True、significant=False、bad_checks=1，验证最低损失更新与早停参考阈值分开管理。

第1题回答“不应该”正确：patience=2，当前仅连续一次未达到改善要求时，bad_checks=1<2，继续等待下一次验证。第2题回答“加载200步”正确：即使在1000步触发停止，已看到的最低损失属于200步，应加载那个文件。两个具体规则例子已答对，不自动推断其掌握所有早停策略。

补充理解：最后一组虽然损失从5.9392475降到5.9378747，但降低约0.0013728不足0.005，因此仍计一次未达到改善要求；同时当前分数确实更低，最佳位置仍更新到4787。回放没有真正节省已经完成的第四轮训练，也没有覆盖原权重文件。

脚本语法树与参考代码一致，仅清理末尾空白，非空代码行仍保留中文注释。只读检查scripts/ci/check_lesson38.py重新回放并核对完整报告，还用不可用记录替代1000步之后的未来，确认patience=1在触及未来前停止且结果不变；检查两份报告字节未变及来源历史未变。代码、报告、学习笔记与CI配置按每课完整上传约定提交，不新增权重。

## 第三十九课：围绕生成文字诊断重复与可读性

用户要求将观察重点转到“能否写出更像样的句子”。本课由学习者新建scripts/learn/23_generation_diagnostics.py，正式报告docs/training_runs/lesson39.json；学习者尚未完成，不提前验收。固定第37课最佳模型，不训练或修改旧生成函数。之前“约4到6课”只是教学安排的估计，不是保证达到通顺生成的承诺；是否改善由实际样例判断。

预先固定开头“春风”“唐僧道：”“宝玉听了”；每段续写60个编号、最近32个编号的上下文；贪心每个开头一次，其他三种采样各用42和43两个种子，总计21段。四种设置为贪心、全词表T=1、全词表T=0.7、top-k=40且T=0.7。T=1对T=0.7只改变温度；T=0.7对top-k40/T=0.7只改变候选限制。相同种子便于复现，但不同采样分布的随机轨迹不意味着逐字一一对应，也没有据两个种子作统计显著性结论。

Top-k：取当前分数最高的k个编号组成S，其他候选概率为0，保留候选的概率按q_i=exp(z_i/T)/Σ_{j∈S}exp(z_j/T)重新归一化。示例原概率[0.6,0.3,0.1]，T=1且k=2，变为[2/3,1/3,0]。k=1退化为贪心。低温或小k可能减少低概率生僻字，也可能加强循环重复，并不保证更通顺。正温度不改变候选分数排名。

重复指标repeat4只统计新增编号，不含开头：长度60产生57个可重叠四元组，repeat4=1−不同四元组数量/57。每种片段首次出现不算重复；后续重复出现才计入分子。例如AAAAAA产生3个AAAA，只有1种，重复比例2/3；ABCDEF产生ABCD、BCDE、CDEF这3种不同四元组，比例0。该指标不判断语义，随机乱码也可能几乎不重复。这里按编号统计，UNK显示为多个字符时也算一个编号，所以不直接称为四个显示汉字。

必须人工检查：是否能接上开头、是否能读懂动作或句意、是否明显重复、人物和内容是否前后矛盾。固定审阅三个开头各自的greedy和topk40_t07/seed42共6段，不通过不断换种子只展示最好看的样本。其他15段也完整保留在报告中。可以把人工观察写进docs/myself_note.md，程序报告不预填“流畅”结论，也不凭repeat4最小自动选优。

本课改变解码后的采样分布，不改变模型权重或原始logits；它不会自动提高此前模型验证交叉熵。生成仍使用最近32字重新计算，与旧代码相同。目标是判断重复与生僻字问题能否靠解码缓解，哪些问题仍需训练或结构改善，不用禁用逗号或屏蔽UNK掩盖问题。

### 第一块：固定对照设置及重复统计

```python
from importlib import import_module  # 复用模型、权重加载和温度概率函数。
from pathlib import Path  # 从脚本位置找到项目目录。
import hashlib  # 检查生成前后权重文件没有变化。
import json  # 读取模型来源并保存诊断记录。
import numpy as np  # 选择候选编号和执行概率采样。

ROOT = Path(__file__).resolve().parents[2]  # 本文件位于scripts/learn。
network = import_module("07_tiny_rnn")  # 使用已经训练好的模型结构。
training = import_module("11_train_steps")  # 复用权重加载函数。
decoding = import_module("12_generate_text")  # 复用稳定的温度概率计算。
PROMPTS = ("春风", "唐僧道：", "宝玉听了")  # 预先固定三个开头，避免只展示最好看的一个。
MODES = (("greedy", None, None), ("sample_t1", 1.0, None), ("sample_t07", 0.7, None), ("topk40_t07", 0.7, 40))  # 分别比较贪心、温度变化及候选限制。


def repeat_rate(ids, n=4):  # 统计新生成编号中扣除首次出现后的重复片段比例。
    assert n >= 1, "片段长度必须为正"  # 本课默认连续四个编号为一个片段。
    pieces = [tuple(ids[start:start + n]) for start in range(len(ids) - n + 1)]  # 用步长一取出可重叠的片段。
    return 0.0 if not pieces else 1.0 - len(set(pieces)) / len(pieces)  # 片段越重复，比例通常越高，但不直接衡量通顺程度。
```

### 第二块：实现top-k采样生成

```python
def generate_text(model, vocab, prompt, temperature=None, top_k=None, seed=42, new_tokens=60):  # 固定最近三十二字上下文，支持候选限制采样。
    tokens = vocab["tokens"]  # 使用权重保存的词表顺序。
    char_to_id = {char: index for index, char in enumerate(tokens)}  # 恢复字符到编号的对应关系。
    assert isinstance(prompt, str) and prompt and isinstance(new_tokens, int) and new_tokens >= 0, "开头或长度无效"  # 至少提供一个输入字。
    assert top_k is None or isinstance(top_k, int) and 1 <= top_k <= len(tokens), "候选数量无效"  # 候选数量不能超出词表。
    assert temperature is None or np.isfinite(temperature) and temperature > 0, "温度无效"  # 采样温度必须为正。
    assert temperature is not None or top_k is None, "贪心模式不需要top-k"  # 本课只在采样模式限制候选。
    history = [char_to_id.get(char, vocab["unk_id"]) for char in prompt]  # 未见输入字符使用固定未知编号。
    generated = []  # 只保存续写编号，不把开头计入重复率。
    rng = np.random.default_rng(seed)  # 随机生成器在整个生成循环中持续使用。
    for _ in range(new_tokens):  # 每次生成一个编号并接回上下文。
        x = np.asarray([history[-32:]], dtype=np.int64)  # 与旧生成函数使用同样的上下文长度。
        scores = model.forward(x)[0, -1, :]  # 最后位置给出下一字的分数。
        if temperature is None:  # 贪心每次直接选最高分。
            next_id = int(np.argmax(scores))  # 不进行随机抽样。
        else:  # 指定温度时才使用概率采样。
            candidates = np.arange(len(tokens)) if top_k is None or top_k == len(tokens) else np.argsort(-scores, kind="stable")[:top_k]  # 全词表或最高分的k个编号作为候选。
            probabilities = decoding.sampling_probabilities(scores[candidates], temperature)  # 只在保留候选内重新归一化。
            next_id = int(rng.choice(candidates, p=probabilities))  # 按候选编号及其对应概率抽样。
        history.append(next_id)  # 新编号会成为下一步的输入。
        generated.append(next_id)  # 单独记录续写部分。
    return prompt + "".join(tokens[index] for index in generated), generated  # 返回完整文字及续写编号。
```

### 第三块：保留所有样例并检查权重不变

```python
def main():  # 固定权重，比较四种解码设置的生成结果。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 使用完整第四轮选出的最佳模型。
    assert source["status"] == "completed", "第四轮尚未完成"  # 不使用未核验训练结果。
    model_path = ROOT / "model" / source["best_checkpoint"]  # 指向保存的最佳权重。
    before_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()  # 留下原权重文件校验值。
    model, vocab, step = training.load_checkpoint(model_path, network)  # 只加载，不训练。
    assert vocab["train_sha256"] == source["train_sha256"], "词表来源不一致"  # 保证输入编号含义正确。
    saved = {name: value.copy() for name, value in model.params.items()}  # 结束时检查内存参数未变。
    rows = []  # 保存所有结果，不只保留看起来最好的一段。
    for prompt in PROMPTS:  # 每种设置都使用同样的三个开头。
        for mode, temperature, top_k in MODES:  # 成对比较温度和top-k的影响。
            for seed in ((42,) if temperature is None else (42, 43)):  # 贪心不需要多个种子，采样固定两个种子观察波动。
                text, ids = generate_text(model, vocab, prompt, temperature, top_k, seed)  # 每段固定续写六十个编号。
                assert len(ids) == 60 and all(0 <= index < len(vocab["tokens"]) for index in ids), "生成编号无效"  # 检查生成长度及范围。
                row = {"prompt": prompt, "mode": mode, "temperature": temperature, "top_k": top_k, "seed": seed, "text": text, "generated_ids": ids, "repeat4": repeat_rate(ids), "unknown_count": ids.count(vocab["unk_id"])}  # 记录解码条件、全文及辅助统计。
                rows.append(row)  # 全量保留，避免只挑选好看的例子。
                print(f"开头={prompt!r} mode={mode} seed={seed} repeat4={row['repeat4']:.1%}", flush=True)  # 显示重复片段比例。
                print(repr(text), flush=True)  # 用repr保留换行及特殊字符。
    summaries = []  # 汇总各设置在这些固定样例上的平均重复比例。
    for mode, _, _ in MODES:  # 每种设置单独汇总。
        group = [row for row in rows if row["mode"] == mode]  # 收集对应三个开头及预定种子的结果。
        average = sum(row["repeat4"] for row in group) / len(group)  # 各段续写长度相同，按样例平均。
        summaries.append({"mode": mode, "samples": len(group), "mean_repeat4": average})  # 重复统计不能当作语言质量分数。
        print(f"汇总 {mode}: samples={len(group)} mean_repeat4={average:.1%}", flush=True)  # 先看统计，再人工阅读文字。
    old_text, old_ids = decoding.generate(model, vocab, PROMPTS[0], new_tokens=60)  # 核对新函数没有改变原贪心行为。
    assert rows[0]["text"] == old_text and rows[0]["generated_ids"] == old_ids, "贪心实现与旧版不一致"  # 新增采样功能不应破坏贪心。
    first_text, first_ids = generate_text(model, vocab, PROMPTS[0], 0.7, 40, 42)  # 再次用相同种子生成top-k样例。
    repeated_text, repeated_ids = generate_text(model, vocab, PROMPTS[0], 0.7, 40, 42)  # 核对当前环境中可复现。
    assert first_text == repeated_text and first_ids == repeated_ids, "固定种子未复现"  # 不比较不同平台的整段黄金文字。
    assert all(np.array_equal(value, saved[name]) for name, value in model.params.items()) and hashlib.sha256(model_path.read_bytes()).hexdigest() == before_hash, "生成修改了权重"  # 内存参数和磁盘文件都应不变。
    return {"checkpoint": model_path.name, "checkpoint_sha256": before_hash, "completed_steps": step, "train_sha256": vocab["train_sha256"], "config": {"new_tokens": 60, "context_size": 32, "sampling_seeds": [42, 43], "repeat_ngram": 4}, "rows": rows, "summaries": summaries}  # 保存全部原始结果，人工评价另行记录。
```

### 第四块：保存正式报告

```python
if __name__ == "__main__":  # 直接运行时才生成诊断报告。
    result = main()  # 本课只前向生成，不进行训练。
    path = ROOT / "docs/training_runs/lesson39.json"  # 保存独立生成对照报告。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存全部设置、统计和文本。
    print("生成诊断报告：", path)  # 输出结果位置。
```

### 实操任务与思考

四块依次放在同一文件，第一二三块的函数定义均在顶层，第四块为脚本入口。运行python scripts/learn/23_generation_diagnostics.py，不重新训练。请提交四条汇总行，以及预先指定6段文字的观察：比贪心是否更少重复、是否真的更容易读懂。问题一：top-k=1时能否随机选到最高分以外的编号？问题二：一段乱码没有重复，是否就能说明它更通顺？老师检查后再完整上传脚本、报告和人工观察，不新增权重。

### 老师预跑与人工阅读

参考代码已在内存执行，没有替学习者创建脚本或正式报告。旧贪心一致、固定种子复现、权重内存和文件不变、top-k=1退化贪心及重复统计示例检查通过。21段均保留在内存预跑结果，本节记录汇总及预先指定6段供人工检查。

| 模式 | 样例数 | 平均repeat4 |
|---|---:|---:|
| greedy | 3 | 84.210526% |
| sample_t1 | 6 | 0.000000% |
| sample_t07 | 6 | 0.000000% |
| topk40_t07 | 6 | 1.169591% |

开头'春风'，模式greedy，种子42，repeat4=0.877193：

```text
春风，只是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人
```

开头'春风'，模式topk40_t07，种子42，repeat4=0.0：

```text
春风一个了。”如此了，我见也去了。”我人得有两个事？”那者道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“
```

开头'唐僧道：'，模式greedy，种子42，repeat4=0.877193：

```text
唐僧道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人
```

开头'唐僧道：'，模式topk40_t07，种子42，repeat4=0.017544：

```text
唐僧道：“我日有个好去。”一个家：“这个里，见个了人大家。”一个道：“哥里，一个人不是道：“不个那，便了道：“你了的子。”我是道
```

开头'宝玉听了'，模式greedy，种子42，repeat4=0.77193：

```text
宝玉听了。”那个道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我
```

开头'宝玉听了'，模式topk40_t07，种子42，repeat4=0.0：

```text
宝玉听了，一个在个大上。”这个要上人。”说人见一个来头？”那人道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“
```

老师对固定6段的阅读结论：贪心在三个开头都进入明显长循环；top-k/T=0.7可打断该长循环，但仍出现“一个在个大上”“这个人人了”等难懂或不合语法的片段。三个开头未表现出稳定的角色和事件连贯性，不能评为已能流畅续写。因此本次诊断支持继续改进模型训练，而不是靠调整解码宣告生成质量达标。

### 学习者第三十九课实操验收与生成诊断

学习者已提交21段完整生成结果和lesson39.json。greedy的平均repeat4为0.8421052631578947（84.2%），sample_t1和sample_t07均为0，topk40_t07为0.011695906432748574（1.2%）。开头、模式、种子、续写长度和第四轮最佳权重来源均符合预定设置。脚本语法树与已预跑参考代码一致，仅清理行末空白，保持每个非空代码行的中文注释。

第1题回答“不能”正确：top-k=1只剩一个最高分候选，其采样概率为1，不能选到分数更低的编号；最高分并列时，本实现的稳定排序保留其中一个编号，与np.argmax的首项选择一致。第2题回答“不能”正确：不重复只表示该长度的编号片段未重复，不表示文字语法或意思正确。

学习者未另给逐段人工阅读评价。以下为老师对预先指定六段（各开头的贪心与top-k/T=0.7种子42）的观察：三个贪心样例均进入“不是我人”等长循环，续写缺少有意义发展；三个top-k样例确实打断长循环，但“春风一个了”“一个在个大上”“这个人人了”等片段难以理解，不能判为通顺。对话标点和“一个、道、这个”等局部形式存在，不足以构成连贯的人物动作或事件。

本次诊断结论：解码方法能缓解明显循环，但没有解决主要语言组织问题。不能据repeat4为0认定普通采样最好，也不把同一模型换解码说成模型本身验证损失改善。下一步应回到训练优化和必要的结构改进，不继续只靠换种子或限制候选追求好看的单个输出。

新增只读检查scripts/ci/check_lesson39.py：检查21个组合完整、编号与文本解码一致，独立用片段计数复算重复率及汇总；沿已保存轨迹核对贪心和top-k编号的排名约束，容许微小平台误差；重跑本机固定种子复现、旧贪心兼容、top-k=1退化为贪心及权重和报告未变检查。跨平台不将完整采样字符串作为黄金值。按每课完整上传约定提交代码、报告、笔记和CI配置，不新增权重。

## 第四十课：实现Adam，并做实际续训对照

用户明确要求从生成诊断回到训练优化。本课由学习者新建scripts/learn/24_adam_experiment.py，输出lesson40_sgd.npz、lesson40_adam.npz及docs/training_runs/lesson40.json；学习者尚未实操，不提前验收。模型仍为719723个参数，从第37课最佳权重累计19148步独立分出两组；相同种子46的前200批、相同裁剪阈值1，SGD学习率0.03、Adam学习率0.001。它比较两个预先指定的优化配置，并不是只更换优化器而保持学习率相同，也没有给两种方法分别做最优超参数搜索；不能推出Adam在所有任务中更好。

原始算法核对来源：[Kingma与Ba，Adam: A Method for Stochastic Optimization，Algorithm 1](https://arxiv.org/pdf/1412.6980)。本课采用m_t=β1m_{t−1}+(1−β1)g_t，v_t=β2v_{t−1}+(1−β2)g_t²；m_hat=m_t/(1−β1^t)，v_hat=v_t/(1−β2^t)；θ←θ−ηm_hat/(sqrt(v_hat)+ε)。数组乘法、平方、开根和除法都是逐元素；v是平方的移动平均，不是减去均值后的方差。默认β1=0.9、β2=0.999、ε=1e−8，ε放在平方根外。

m用于综合历史梯度方向，v用于描述历史平方尺度。零初始化的移动平均在早期偏小，因此使用1−β^t作偏差修正。第一步g=0.4时m=0.04、v=0.00016，修正后m_hat=0.4、v_hat=0.16，学习率0.001使参数1.0变为约0.999。Adam仍然需要选择学习率，不保证每步损失下降，也不保证更新总与当前单批负梯度同向，因为m包含历史。

新建Adam时m=v=0且t=0，模型此前的19148次SGD更新不能冒充已有Adam状态。每个批次t只增加一次，五组参数共同使用这个t，不是在每组参数更新时各加一次。每组Adam仅在批次循环外创建一次，不能每批重建，否则梯度历史丢失。本次Adam历史最终t=200，而模型累计步数是19348。

先求完整梯度，再做全局范数裁剪，最后交给Adam。SGD中的更新长度等于η乘裁剪后梯度长度，但Adam会用历史状态缩放每个分量，不能套用同一长度结论。额外m和v有2×719723个float32数，共5757784字节，约5.49MiB；不是模型新增可学习参数，也不代表整个训练进程内存。

本课只保存用于加载评估的权重，报告明确optimizer_state_saved=False。精确恢复Adam续训还需要m、v、t和训练数据位置等状态，仅加载权重后新建Adam会重新开始优化器历史；当前文件不支持精确断点续训。生成设置固定为3个旧开头、贪心及top-k40/T=0.7/seed42、各60个续写编号，起点和两个分支各保留6段，总共18段，不能只看损失或挑一个好例子。

### 第一块：导入及预先规定的配置

```python
from importlib import import_module  # 复用数据、模型、检查点和生成诊断。
from itertools import islice  # 固定每组训练两百个批次。
from pathlib import Path  # 定位项目文件及实验输出。
import hashlib  # 检查共同起点权重没有变化。
import json  # 读取旧报告并保存新实验。
import random  # 固定训练章节顺序。
import numpy as np  # 保存优化器状态并执行逐元素计算。

ROOT = Path(__file__).resolve().parents[2]  # 当前脚本位于scripts/learn。
network = import_module("07_tiny_rnn")  # 模型结构保持不变。
training = import_module("11_train_steps")  # 复用权重加载和保存。
epoch_tools = import_module("17_train_epochs")  # 复用完整验证函数。
diag = import_module("23_generation_diagnostics")  # 保持上一课的生成和重复统计规则。
STEPS = 200  # 每组预先固定两百次更新。
CONFIGS = (("sgd", 0.03), ("adam", 0.001))  # 比较这两个指定配置，不宣称学习率已经分别调到最优。
```

### 第二块：Adam状态、更新和手算核对

```python
class Adam:  # 为五组参数分别保存梯度历史。
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):  # 创建一次优化器，随后跨批次复用。
        assert np.isfinite(learning_rate) and learning_rate > 0 and np.isfinite(epsilon) and epsilon > 0, "学习率或稳定项无效"  # 两个数都必须为有限正数。
        assert 0 <= beta1 < 1 and 0 <= beta2 < 1, "历史保留系数无效"  # 系数控制指数移动平均。
        self.lr, self.beta1, self.beta2, self.epsilon = learning_rate, beta1, beta2, epsilon  # 保存优化器超参数。
        self.m = {name: np.zeros_like(value) for name, value in params.items()}  # m保存梯度的指数移动平均。
        self.v = {name: np.zeros_like(value) for name, value in params.items()}  # v保存梯度平方的指数移动平均。
        self.t = 0  # 新建Adam的历史从零开始，与模型累计训练步数分开。

    def step(self, params, grads):  # 用本批完整梯度更新所有参数。
        assert params.keys() == grads.keys() == self.m.keys(), "参数名称不一致"  # 每组参数必须有对应梯度和状态。
        self.t += 1  # 每个批次只增加一次，不能在参数循环里增加。
        for name in params:  # 每个状态数组与对应参数逐元素对齐。
            g = grads[name]  # 本课传入已经完成全局裁剪的梯度。
            assert g.shape == params[name].shape and np.isfinite(g).all(), "梯度形状或数值无效"  # 避免错误广播或无效更新。
            self.m[name] *= self.beta1  # 保留过去梯度平均的一部分。
            self.m[name] += (1 - self.beta1) * g  # 加入当前梯度。
            self.v[name] *= self.beta2  # 保留过去梯度平方平均的一部分。
            self.v[name] += (1 - self.beta2) * g * g  # 加入当前梯度的逐元素平方。
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)  # 修正从零初始化造成的偏小。
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)  # 对平方平均做同样的初始化偏差修正。
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  # 用历史方向和平方尺度调整每个参数的更新。


def check_first_step():  # 用能手算的小例子验证公式和批次计数。
    params = {"weight": np.array([1.0]), "zero": np.array([2.0])}  # 两组双精度参数便于核对。
    optimizer = Adam(params)  # 默认学习率0.001，两组状态均从零开始。
    optimizer.step(params, {"weight": np.array([0.4]), "zero": np.array([0.0])})  # 一次更新同时处理两个参数组。
    expected = 1.0 - 0.001 * 0.4 / (0.4 + 1e-8)  # 第一步修正后m为0.4，v为0.16。
    assert optimizer.t == 1 and np.isclose(params["weight"][0], expected, rtol=0, atol=1e-12) and params["zero"][0] == 2.0, "Adam首步公式或计数错误"  # t只增一次，零历史零梯度不移动。
    print("Adam手算首步参数：", params["weight"][0], "优化器步数：", optimizer.t, flush=True)  # 预期约0.999和一步。
```

### 第三块：同一完整验证与固定生成样例

```python
def measure(model, vocab, label, data, validation):  # 对各候选使用同一验证和生成规则。
    loss, positions = epoch_tools.evaluate_full(model, data, validation)  # 固定参数评估完整验证集。
    assert positions == 316960, "验证范围不一致"  # 不混用部分验证结果。
    print(f"{label} full_validation={loss:.6f}", flush=True)  # 完整验证损失用于候选选择。
    samples = []  # 额外保留固定开头的实际文字。
    for prompt in diag.PROMPTS:  # 沿用春风、唐僧道和宝玉听了三个开头。
        for mode, temperature, top_k in (("greedy", None, None), ("topk40_t07", 0.7, 40)):  # 对每个候选都使用同两种解码设置。
            text, ids = diag.generate_text(model, vocab, prompt, temperature, top_k, 42)  # 固定六十个续写编号和种子42。
            samples.append({"prompt": prompt, "mode": mode, "seed": 42, "text": text, "generated_ids": ids, "repeat4": diag.repeat_rate(ids)})  # 全部保留，避免挑选好看的样例。
            print(label, mode, repr(text), flush=True)  # 指标下降时也检查句子是否更容易读懂。
    return {"validation_loss": loss, "validation_positions": positions, "samples": samples}  # 返回当前候选的完整评估记录。
```

### 第四块：共同起点与训练前缀

```python
def main(output_dir=None):  # 同起点、同数据、同更新预算比较两种优化配置。
    check_first_step()  # 先确认Adam的基本公式无误。
    data = import_module("06_make_windows")  # 复用已划分的数据和固定词表。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 从完整第四轮最佳模型开始。
    assert source["status"] == "completed" and source["train_sha256"] == data.vocab["train_sha256"], "模型来源或数据版本不一致"  # 不使用其他实验起点。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 两组都加载同一最佳文件。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 保留共同起点校验值。
    initial, vocab, start_step = training.load_checkpoint(source_path, network)  # 用原模型作为共同起点和候选。
    assert vocab == data.vocab and start_step == 19148, "起点配置不正确"  # 本次从已核验的第四轮结束位置出发。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 老师预跑可把分支文件放入临时目录。
    records = list(data.train_records)  # 从原始章节列表复制。
    random.Random(46).shuffle(records)  # 预先固定本次短训练顺序。
    batches = list(islice(data.iter_batches(data.iter_windows(records, 32, 32), 16), STEPS))  # 两组精确复用相同的两百批。
    assert len(batches) == STEPS and sum(batch["y"].size for batch in batches) == 102400, "训练预算错误"  # 每组新增相同数量的目标预测。
    for batch in batches:  # 缓存输入目标不允许被训练代码修改。
        batch["x"].setflags(write=False)  # 保留输入数组。
        batch["y"].setflags(write=False)  # 保留正确目标数组。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 准备统计训练前缀实际覆盖。
    book_windows = {record["book"]: 0 for record in records}  # 没覆盖到的书保留零计数。
    for batch in batches:  # 每行对应一个章节内窗口。
        for chapter_id in batch["chapter_ids"]:  # 按窗口来源统计。
            book_windows[chapter_to_book[chapter_id]] += 1  # 不假定短前缀四书均衡。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 两组使用同一完整验证集。
    initial_row = measure(initial, vocab, "starting_model", data, validation)  # 起点也参加选优。
    assert np.isclose(initial_row["validation_loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点指标不一致"  # 确认未加载错模型。
    initial_row.update({"label": "starting_model", "checkpoint": source_path.name, "completed_steps": start_step})  # 加入来源信息。
    history = [initial_row]  # 保留全部候选，避免两个分支都变差时丢掉旧模型。
```

### 第五块：两组独立训练、选优和报告入口

```python
    for label, learning_rate in CONFIGS:  # 每组是独立训练分支。
        model, current_vocab, loaded_step = training.load_checkpoint(source_path, network)  # 每组重新加载共同起点。
        assert current_vocab == vocab and loaded_step == start_step and all(np.array_equal(value, initial.params[name]) for name, value in model.params.items()), "分支起点不同"  # 第二组不能接着第一组训练。
        optimizer = Adam(model.params, learning_rate) if label == "adam" else None  # Adam只创建一次，在组内所有批次间保留历史。
        for local_step, batch in enumerate(batches, start=1):  # 本组连续完成两百次更新。
            loss, raw_grads = model.loss_and_grad(batch["x"], batch["y"])  # 使用同一模型结构和反向公式。
            assert np.isfinite(loss), "训练损失无效"  # 无效数值不继续更新。
            grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 两组都先做同样的全局梯度裁剪。
            if optimizer is None:  # SGD使用固定学习率乘当前梯度。
                for name in model.params:  # 完整反向后统一更新。
                    model.params[name] -= learning_rate * grads[name]  # 不保存梯度历史。
            else:  # Adam使用过去和当前梯度形成的状态。
                optimizer.step(model.params, grads)  # 同一个优化器对象持续累积m、v和t。
            assert all(np.isfinite(value).all() for value in model.params.values()), "参数更新异常"  # 两组都检查数值稳定。
            if local_step % 50 == 0:  # 每五十步显示训练进度。
                print(f"{label} step={local_step}/{STEPS} batch_loss={loss:.6f}", flush=True)  # 当前批次损失不用于最终模型选择。
        assert optimizer is None or optimizer.t == STEPS, "Adam历史计数错误"  # Adam本次历史为200步，模型累计步数是19348。
        row = measure(model, vocab, label, data, validation)  # 训练结束后评估固定样例和完整验证。
        path = output_dir / f"lesson40_{label}.npz"  # 两组保存独立模型文件。
        training.save_checkpoint(path, model, vocab, start_step + STEPS)  # 本课只保存模型权重，不包含Adam状态。
        state_bytes = 0 if optimizer is None else sum(value.nbytes for group in (optimizer.m, optimizer.v) for value in group.values())  # 统计额外m和v数组大小。
        row.update({"label": label, "learning_rate": learning_rate, "checkpoint": path.name, "completed_steps": start_step + STEPS, "change_from_start": row["validation_loss"] - initial_row["validation_loss"], "optimizer_steps": STEPS, "adam_state_steps": None if optimizer is None else optimizer.t, "optimizer_state_bytes": state_bytes})  # 记录性能变化和优化器额外状态。
        history.append(row)  # 无论结果好坏都保留。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash and all(np.array_equal(value, initial.params[name]) for name, value in training.load_checkpoint(source_path, network)[0].params.items()), "共同起点被修改"  # 文件和初始模型对象都应保持不变。
    selected = min(history, key=lambda row: row["validation_loss"])  # 在起点和两组最终候选中选最低损失。
    print("实际训练窗口覆盖：", book_windows, flush=True)  # 明确本次只是相同短前缀。
    print("本次选中：", selected["label"], selected["validation_loss"], flush=True)  # 不预设Adam必须胜出。
    return {"starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"], "config": {"steps_per_branch": STEPS, "sgd_lr": 0.03, "adam_lr": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "max_norm": 1.0, "batch_size": 16, "seq_length": 32, "chapter_shuffle_seed": 46}, "training_positions_per_branch": 102400, "book_windows": book_windows, "optimizer_state_saved": False, "history": history, "selected_label": selected["label"], "selected_checkpoint": selected["checkpoint"]}  # 保存对照条件，不声称权重文件支持精确恢复Adam训练。


if __name__ == "__main__":  # 直接运行才训练并保存正式报告。
    result = main()  # 默认把分支权重保存到model。
    path = ROOT / "docs/training_runs/lesson40.json"  # 使用独立的实验记录。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保报告目录存在。
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存配置、指标和所有固定样例。
    print("Adam对照报告：", path)  # 输出结果位置。
```

### 实操任务

第一、二、三块都是顶层导入或类/函数；第四块定义main，第五块继续main四格缩进，末尾if入口回到顶层。运行python scripts/learn/24_adam_experiment.py。两组共400次更新，加上三次完整验证，可能需要一段时间。请提交起点与两个分支完整验证损失、最后选中项，以及固定生成样例是否更好读。问题一：能否每个批次重新创建Adam对象？问题二：本课只保存权重，没有m/v/t，能否用它精确恢复中断的Adam训练？学习者完成后再核验、添加两个权重明确例外并完整上传。

### 老师完整预跑

只在临时目录保存分支权重，完成后已清理，没有创建学习者练习脚本或正式实验报告。手算首步、跨批历史计数、共同起点不变检查通过。实际书籍窗口覆盖：{"三国演义.txt": 412, "西游记.txt": 1117, "水浒全传.txt": 611, "红楼梦.txt": 1060}。

| 候选 | 完整验证损失 | 累计步数 |
|---|---:|---:|
| starting_model | 5.937874744 | 19148 |
| sgd | 5.970440074 | 19348 |
| adam | 5.980868139 | 19348 |

本次选中starting_model；临时模型文件大小：{"lesson40_adam.npz": 2682590, "lesson40_sgd.npz": 2687831}字节。完整配置与所有固定文本预跑记录如下，不用单个样例保证模型已经流畅：

```json
{
  "starting_checkpoint": "lesson37_best.npz",
  "starting_sha256": "9d306c2bc19218748c19ccf96d2f97706f16c2e7180fa9013007661b732e97c6",
  "train_sha256": "0cee762b4832c3a1c37c84c245ca912c51a1c22fa8df4c37e811ec735e390a69",
  "config": {
    "steps_per_branch": 200,
    "sgd_lr": 0.03,
    "adam_lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "max_norm": 1.0,
    "batch_size": 16,
    "seq_length": 32,
    "chapter_shuffle_seed": 46
  },
  "training_positions_per_branch": 102400,
  "book_windows": {
    "三国演义.txt": 412,
    "西游记.txt": 1117,
    "水浒全传.txt": 611,
    "红楼梦.txt": 1060
  },
  "optimizer_state_saved": false,
  "history": [
    {
      "validation_loss": 5.937874743698944,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，只是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风一个了。”如此了，我见也去了。”我人得有两个事？”那者道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“我日有个好去。”一个家：“这个里，见个了人大家。”一个道：“哥里，一个人不是道：“不个那，便了道：“你了的子。”我是道",
          "repeat4": 0.01754385964912286
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了。”那个道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我",
          "repeat4": 0.7719298245614035
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了，一个在个大上。”这个要上人。”说人见一个来头？”那人道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“",
          "repeat4": 0.0
        }
      ],
      "label": "starting_model",
      "checkpoint": "lesson37_best.npz",
      "completed_steps": 19148
    },
    {
      "validation_loss": 5.970440073894287,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风。”如王道：“将不不是那下去，一个一见，以了山日，只有一个人见马，便人，只见不里，就来，那来一个，不得道：“有了，都道：",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：",
          "repeat4": 0.8947368421052632
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“吾此将，方着马，我了军大的，只在一人，大不中出之，也不是大得兵，便有道：“你只一知之来。\n今是那人，不听上。\n日说：“",
          "repeat4": 0.0
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了，不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“我不是道：“",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了。”老明道：“若不人，与他日之，我不听不不去。\n老曰：“我下者也，一个，不知道：“却便一知之了，说：“我两见军得，将道：",
          "repeat4": 0.0
        }
      ],
      "label": "sgd",
      "learning_rate": 0.03,
      "checkpoint": "lesson40_sgd.npz",
      "completed_steps": 19348,
      "change_from_start": 0.03256533019534302,
      "optimizer_steps": 200,
      "optimizer_state_bytes": 0,
      "adam_state_steps": null
    },
    {
      "validation_loss": 5.980868139271782,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，只不道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道",
          "repeat4": 0.7017543859649122
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风之。我了一回，可道：“便将的不得，又然一里，即到说：“我是有下尚，便是一个一个不知。\n者兵大得，何道：“者人之着，如了一",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得",
          "repeat4": 0.7543859649122807
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“你去。”正明道：“这又笑道：“不是大子，又叫他得，都道：“宝过，便道：“你见老了道：“众中，一道：“我在僧将之、，不是",
          "repeat4": 0.0
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了，只不得道：“我不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一",
          "repeat4": 0.7017543859649122
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了。”二明道：“者道：“小知马道：“我下的子，到此马，便是一个的都道：“我不玉道：“是又，把上不得道：“我宝，是上人一个，",
          "repeat4": 0.03508771929824561
        }
      ],
      "label": "adam",
      "learning_rate": 0.001,
      "checkpoint": "lesson40_adam.npz",
      "completed_steps": 19348,
      "change_from_start": 0.042993395572837656,
      "optimizer_steps": 200,
      "optimizer_state_bytes": 5757784,
      "adam_state_steps": 200
    }
  ],
  "selected_label": "starting_model",
  "selected_checkpoint": "lesson37_best.npz"
}
```

老师对18段固定生成样例的人工观察：SGD和Adam分支均未摆脱语义不清及长循环，Adam的top-k续写虽有“你去”等可读短片段，但后续人物与语句仍不连贯。本次两个分支完整验证损失都高于起点，因此保留starting_model。结论仅限这两个预设配置、同一200批前缀的短期续训，不证明Adam普遍不如SGD；下一次实验应另行预定更充分的数据覆盖和预算，不在看到结果后悄悄延长某一分支。

额外用连续梯度0.4、0.2核对第二步历史状态：m2=0.056、v2=0.00019984，按第二步偏差修正得到的参数与Adam实现一致，t=2检查通过。报告元数据把optimizer_steps明确记为两组各200步，并单独记录adam_state_steps（SGD为null、Adam为200），避免将SGD未维护Adam状态误解为未训练。此处仅调整报告字段，不改变已执行的训练计算。

### 学习者第四十课验收与回答

学习者提交完整运行输出及lesson40.json。起点完整验证损失5.937874743698944，SGD分支5.970440073894287，Adam分支5.980868139271782；两组变化分别+0.03256533019534302和+0.042993395572837656，因此保留starting_model正确。相同200批前缀共102400目标、3200窗口，书籍覆盖为三国412、西游1117、水浒611、红楼1060。两组均累计19348步，Adam单独状态t为200，m/v额外数组5757784字节。

第1题回答“不能”正确：每批重新创建Adam会清空m、v并把t重置为零，失去跨批次历史，无法按本课Adam定义持续更新。第2题回答“不能”正确：模型权重仅决定当前前向计算，精确恢复Adam轨迹还需要对应m/v/t、超参数、数据迭代位置及必要随机状态。本课明确只保存权重，不声称完整断点恢复。两个具体问题已答对，不据此认定掌握全部偏差修正推导。

学习者未另给逐段人工评价；老师根据18段固定输出判断，两个分支仍有“我不是道”“我不得，一个一个”等循环，top-k样例仍有明显语义不连贯，本次未达到更流畅生成的目标。保留旧模型，不因为算法名称或某个短片段好看而宣告提升。两个不同学习率配置的短期结果不能推广为优化器优劣排序。

脚本语法树与完整预跑参考代码一致，只清理末尾空白，所有非空代码行保留中文注释。新增check_lesson40.py核对首步及连续变化梯度的历史累积、一次批次计数，以及非零历史下当前零梯度仍可能移动；只读加载三份模型复算完整验证、重建训练预算和书籍覆盖、核对保存样例编号及额外状态说明，不重训400步。按每课完整上传约定提交代码、正式报告、两个分支权重、笔记、模型说明和CI配置。用户个人笔记新增语料来源链接一并原样保留，仅去掉文件末尾多余空行，不表示这些来源已经过采集或许可核验。

## 第四十一课：覆盖完整训练集的Adam续训

用户确认做完整训练集Adam实验。由学习者新建scripts/learn/25_train_adam_epoch.py，正式报告docs/training_runs/lesson41.json；尚未实操，不提前验收。继续使用既有四书划分及5547项词表，不混入新增data/LightNovel5000_v1_txt目录，也不将新语料目录纳入本课提交。

从第四轮最佳lesson37_best.npz的累计19148步开始，不从lesson40_adam.npz再从头遍历一轮，以免重复前200批。新建一次Adam，学习率0.001、beta1=0.9、beta2=0.999、epsilon=1e−8；章节种子46、窗口32、批大小16、裁剪阈值1，均与上一课Adam分支相同。重新训练前200步核对指标及参数复现后，继续同一模型、同一生成器和同一Adam历史完成剩余4587步。本轮内学习率不变。

全轮4787次更新，累计19148+4787=23935步，而Adam本次历史t=4787。原四轮是SGD训练，不能把那些更新当作Adam已有历史。每个验证检查只前向，不重新创建优化器、不清空m/v、不把训练模型退回最佳检查点。最佳仅保存独立副本；若要退回旧模型继续Adam，还需要匹配那个位置的优化器状态。

预定完整验证候选0、200、1000、3000、4787，均覆盖316960个预测位置；每个候选保留三个固定开头的贪心及top-k40/T=0.7/seed42各60编号，共六段，全部候选30段。起点也参与选择，严格改善才更新最佳文件。不是完整SGD与Adam同预算对照，因为本课没有额外完整SGD分支；它检验这个Adam配置覆盖整轮后的轨迹，不能单独据此断言优化器普遍优劣。

预期完整覆盖76584个窗口、2450688个目标，尾批8行，模型可学习参数仍719723。m/v额外数组5757784字节，贯穿整轮。当前文件仅保存模型和既有metadata，不保存m/v/t、数据迭代位置，optimizer_state_saved=False。中断后虽有检查点报告，直接重跑仍从旧起点开始，不是自动断点恢复。run_seconds包含起点和周期完整验证、文字生成、保存及核对，不能当作纯训练吞吐时间。

### 第一块：复用上一课工具

```python
from importlib import import_module  # 复用已经完成的Adam和评估工具。
from pathlib import Path  # 定位项目及独立输出文件。
from time import perf_counter  # 记录含训练、验证及保存的总时间。
import hashlib  # 检查旧模型文件没有被覆盖。
import json  # 读取实验来源并保存报告。
import random  # 固定这一轮的训练章节顺序。
import numpy as np  # 检查参数、损失及前缀复现。

ROOT = Path(__file__).resolve().parents[2]  # 当前文件位于scripts/learn。
adam_tools = import_module("24_adam_experiment")  # 复用Adam类与完整验证和固定样例评估。
network = adam_tools.network  # 保持同一RNN结构。
training = adam_tools.training  # 复用模型保存和加载。
epoch_tools = adam_tools.epoch_tools  # 复用报告保存工具。
LEARNING_RATE = 0.001  # 沿用上一课Adam分支的学习率。
```

### 第二块：验证检查点不重建Adam

```python
def record_checkpoint(model, vocab, optimizer, data, validation, local_step, start_step, output_dir):  # 保存并评估本轮指定位置。
    assert optimizer.t == local_step, "Adam历史计数不匹配"  # 验证期间不能重建优化器或额外更新。
    row = adam_tools.measure(model, vocab, f"adam_step{local_step:04d}", data, validation)  # 完整验证并保留六段固定生成样例。
    path = output_dir / f"lesson41_step{local_step:04d}.npz"  # 文件名记录本轮局部步数。
    training.save_checkpoint(path, model, vocab, start_step + local_step)  # 权重元数据记录模型累计更新次数。
    assert optimizer.t == local_step, "评估改变了Adam计数"  # 只做前向评估不应更新优化器。
    row.update({"local_step": local_step, "global_step": start_step + local_step, "adam_steps": optimizer.t, "checkpoint": path.name})  # 三种计数的含义分别记录。
    return row  # 不保存m和v，当前检查点用于评估及模型选择。
```

### 第三块：完整数据与新Adam历史

```python
def main(output_dir=None, report_path=None):  # 完整训练一轮，默认保存正式产物。
    started = perf_counter()  # 耗时包含起点评估、训练、周期验证、生成和保存。
    data = import_module("06_make_windows")  # 沿用原四书训练窗口和词表。
    source = json.loads((ROOT / "docs/training_runs/lesson37.json").read_text(encoding="utf-8"))  # 从完整第四轮最佳模型开始。
    short = json.loads((ROOT / "docs/training_runs/lesson40.json").read_text(encoding="utf-8"))  # 用上一课Adam短分支核对前缀。
    assert source["status"] == "completed" and source["train_sha256"] == short["train_sha256"] == data.vocab["train_sha256"], "来源或数据版本不一致"  # 新语料不自动混入本次对照。
    source_path = ROOT / "model" / source["best_checkpoint"]  # 不加载已经额外训练两百步的Adam分支。
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()  # 留下共同起点校验值。
    assert source_hash == short["starting_sha256"] and short["config"]["adam_lr"] == LEARNING_RATE and short["config"]["chapter_shuffle_seed"] == 46, "短实验条件不一致"  # 前缀复现需要同一起点及设置。
    model, vocab, start_step = training.load_checkpoint(source_path, network)  # 训练模型只加载一次。
    assert vocab == data.vocab and start_step == 19148, "起点模型配置错误"  # 原模型已经训练四轮。
    optimizer = adam_tools.Adam(model.params, learning_rate=LEARNING_RATE)  # 新建一次Adam，m/v为零且t为零，随后全程复用。
    output_dir = ROOT / "model" if output_dir is None else Path(output_dir)  # 老师预跑时可改用临时目录。
    report_path = ROOT / "docs/training_runs/lesson41.json" if report_path is None else Path(report_path)  # 独立报告不覆盖旧实验。
    best_path = output_dir / "lesson41_best.npz"  # 独立最佳权重快照。
    records = list(data.train_records)  # 从相同原始章节顺序复制。
    random.Random(46).shuffle(records)  # 与上一课短实验使用相同顺序。
    validation = [json.loads(line) for line in (ROOT / "data/splits/validation.jsonl").read_text(encoding="utf-8").splitlines()]  # 仍使用固定完整验证集。
    expected_books = {}  # 独立根据章节长度计算整轮规模。
    for record in records:  # 统计应有的章节内窗口。
        count = max(0, (len(record["text"]) - 33) // 32 + 1)  # 三十三个字符形成三十二个错位目标。
        expected_books[record["book"]] = expected_books.get(record["book"], 0) + count  # 汇总各书窗口数量。
    expected_windows = sum(expected_books.values())  # 得到完整训练窗口数。
    expected_steps = (expected_windows + 15) // 16  # 保留不足十六行的尾批。
    initial = adam_tools.measure(model, vocab, "starting_model", data, validation)  # 起点也参加候选比较。
    assert np.isclose(initial["validation_loss"], source["best_validation_loss"], rtol=1e-6, atol=1e-6), "起点验证不一致"  # 防止错误权重或验证范围。
    initial.update({"local_step": 0, "global_step": start_step, "adam_steps": 0, "checkpoint": source_path.name})  # 旧模型累计次数不等于新Adam历史次数。
    history, best_row = [initial], initial  # 保存所有预定候选并先保留旧模型。
    training.save_checkpoint(best_path, model, vocab, start_step)  # 后续都变差时仍有正确的最佳文件。
    report = {  # 保存运行状态、配置和完整候选轨迹。
        "status": "running", "starting_checkpoint": source_path.name, "starting_sha256": source_hash, "train_sha256": vocab["train_sha256"],  # 记录起点来源。
        "config": {"epoch": 5, "learning_rate": LEARNING_RATE, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "batch_size": 16, "seq_length": 32, "max_norm": 1.0, "chapter_shuffle_seed": 46},  # 全轮使用预先规定条件。
        "history": history, "completed_local_steps": 0, "best_local_step": 0, "best_validation_loss": initial["validation_loss"], "best_checkpoint": best_path.name,  # 记录当前选优结果。
        "optimizer_state_saved": False,  # 目前文件仅保存模型权重，不能精确恢复中断的Adam。
    }  # 不使用测试集，不启用自动早停。
    epoch_tools.save_report(report_path, report)  # 先保留起点状态。
    chapter_to_book = {record["id"]: record["book"] for record in records}  # 窗口来源映射到书籍。
    book_windows = {book: 0 for book in expected_books}  # 训练覆盖从零累计。
    seen_windows, seen_positions, loss_sum = 0, 0, 0.0  # 分别累计窗口、预测位置和训练过程损失。
    batches = data.iter_batches(data.iter_windows(records, 32, 32), 16)  # 使用完整生成器，不限制为两百批。
```

### 第四块：完整一轮更新与候选选择

```python
    for local_step, batch in enumerate(batches, start=1):  # 完整遍历这一轮的训练窗口。
        x, y = batch["x"], batch["y"]  # 读取当前批次实际数据。
        loss, raw_grads = model.loss_and_grad(x, y)  # 使用当前模型求完整梯度。
        assert np.isfinite(loss), "训练损失无效"  # 无效值不能继续更新。
        grads, _, _ = network.clip_gradients(raw_grads, 1.0)  # 所有参数的梯度统一裁剪。
        optimizer.step(model.params, grads)  # 始终使用循环外创建的同一个Adam对象。
        assert optimizer.t == local_step and all(np.isfinite(value).all() for value in model.params.values()), "Adam计数或参数异常"  # 检查历史未被重置及参数有限。
        seen_windows += x.shape[0]  # 尾批按实际行数计入。
        seen_positions += y.size  # 每个实际预测位置都计入。
        loss_sum += loss * y.size  # 训练过程损失按实际位置加权。
        for chapter_id in batch["chapter_ids"]:  # 按行核对来源覆盖。
            book_windows[chapter_to_book[chapter_id]] += 1  # 每本书的窗口都应完整访问。
        if local_step % 500 == 0:  # 定期显示训练进度。
            print(f"Adam local_step={local_step}/{expected_steps} global_step={start_step+local_step} t={optimizer.t} batch_loss={loss:.6f}", flush=True)  # 同时显示局部、累计和Adam历史计数。
        if local_step in (200, 1000, 3000, expected_steps):  # 只在预先规定的位置做完整验证。
            row = record_checkpoint(model, vocab, optimizer, data, validation, local_step, start_step, output_dir)  # 评估后继续当前状态，不重建优化器。
            if local_step == 200:  # 验证短实验前缀得到复现。
                previous = next(item for item in short["history"] if item["label"] == "adam")  # 取得上一课Adam分支记录。
                previous_model, previous_vocab, previous_step = training.load_checkpoint(ROOT / "model" / previous["checkpoint"], network)  # 只加载作比较，不替换当前训练模型。
                assert previous_vocab == vocab and previous_step == start_step + local_step and np.isclose(row["validation_loss"], previous["validation_loss"], rtol=1e-6, atol=1e-6), "短实验前缀指标不一致"  # 起点和前两百批条件相同。
                assert all(np.allclose(value, previous_model.params[name], rtol=1e-6, atol=1e-6) for name, value in model.params.items()), "短实验前缀参数不一致"  # 同时核对实际五组参数。
            history.append(row)  # 保留每个候选结果，无论好坏。
            if row["validation_loss"] < best_row["validation_loss"]:  # 只有严格更低才替换最佳快照。
                best_row = row  # 最佳不一定是最新检查点。
                training.save_checkpoint(best_path, model, vocab, row["global_step"])  # 只保存副本，不把训练模型退回旧状态。
            report.update({"completed_local_steps": local_step, "best_local_step": best_row["local_step"], "best_validation_loss": best_row["validation_loss"]})  # 同步报告进度。
            epoch_tools.save_report(report_path, report)  # 每个检查点后保存一次。
```

### 第五块：整轮核对、最佳文件重载与入口

```python
    assert local_step == expected_steps == optimizer.t == 4787 and seen_windows == expected_windows == 76584, "整轮更新或窗口数量错误"  # Adam历史应持续覆盖整轮。
    assert seen_positions == 2450688 and x.shape[0] == 8 and book_windows == expected_books, "目标、尾批或书籍覆盖不一致"  # 不遗漏小尾批或某一本书。
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash, "旧起点权重被覆盖"  # 旧模型保留。
    selected = min(history, key=lambda item: item["validation_loss"])  # 独立核对包括起点的最低分。
    assert selected["local_step"] == best_row["local_step"], "最佳候选错误"  # 不把最后一步自动视为最好。
    best_model, best_vocab, best_step = training.load_checkpoint(best_path, network)  # 训练结束后加载实际最佳文件。
    selected_path = source_path if selected["local_step"] == 0 else output_dir / selected["checkpoint"]  # 被选中的可以是旧模型。
    candidate, candidate_vocab, candidate_step = training.load_checkpoint(selected_path, network)  # 加载对应候选作参数对照。
    assert best_vocab == candidate_vocab == vocab and best_step == candidate_step == selected["global_step"] and all(np.array_equal(value, candidate.params[name]) for name, value in best_model.params.items()), "最佳文件与所选候选不一致"  # 保存内容必须与报告对应。
    state_bytes = sum(value.nbytes for group in (optimizer.m, optimizer.v) for value in group.values())  # 统计全程保留的m/v数组额外内存。
    report.update({"status": "completed", "completed_global_steps": start_step + local_step, "adam_steps": optimizer.t, "seen_windows": seen_windows, "seen_positions": seen_positions, "last_batch_size": x.shape[0], "book_windows": book_windows, "training_process_loss": loss_sum / seen_positions, "adam_state_bytes": state_bytes, "run_seconds": perf_counter() - started})  # 记录完整覆盖及含验证保存的运行时间。
    epoch_tools.save_report(report_path, report)  # 全部核对结束后标记完成。
    print("本轮、累计、Adam历史步数：", local_step, report["completed_global_steps"], optimizer.t, flush=True)  # 预期为4787、23935、4787。
    print("最佳局部步数及完整验证损失：", best_row["local_step"], best_row["validation_loss"], flush=True)  # 最佳来自预定五个候选。
    print("Adam整轮覆盖、前缀复现与最佳参数检查：通过", flush=True)  # 不把验证损失当作句子流畅度评分。
    return report  # 返回本轮完整记录。


if __name__ == "__main__":  # 直接运行时才训练并保存正式产物。
    main()  # 默认写入model和docs/training_runs。
```

### 实操任务

五块依次输入。第一二块顶层导入和函数；第三块定义main；第四、五块继续main保持四格缩进，末尾if回到顶层。运行python scripts/learn/25_train_adam_epoch.py。生成lesson41_step0200.npz、lesson41_step1000.npz、lesson41_step3000.npz、lesson41_step4787.npz、lesson41_best.npz及报告lesson41.json。请提交五个候选完整验证损失、最后三种步数、最佳局部步数，以及固定文字是否比起点更容易理解。问题一：完成第200步验证后，应继续使用现有Adam还是重新创建？问题二：若所有新检查点损失都高于起点，应该选哪个？学习者完成后再核验、添加明确权重例外并完整上传，不额外上传新增语料目录。

### 老师完整预跑

临时目录完成整轮及所有检查，临时权重和报告随后清理；没有创建学习者练习脚本或正式产物。

| 局部步数 | 累计步数 | Adam历史t | 完整验证损失 |
|---:|---:|---:|---:|
| 0 | 19148 | 0 | 5.937874744 |
| 200 | 19348 | 200 | 5.980868139 |
| 1000 | 20148 | 1000 | 5.808636784 |
| 3000 | 22148 | 3000 | 5.366861238 |
| 4787 | 23935 | 4787 | 5.118321193 |

最佳局部步数4787，最佳完整验证损失5.118321193317643。实际训练覆盖76584窗口、2450688目标、尾批8行；Adam历史最终4787步，含验证保存的总耗时约720.3秒。临时文件大小：{"lesson41_best.npz": 2677547, "lesson41_step0200.npz": 2682590, "lesson41_step1000.npz": 2678160, "lesson41_step3000.npz": 2678330, "lesson41_step4787.npz": 2677547}字节。

以下省略样例编号，只保留全部固定文字和指标，正式实操报告会保留编号：

```json
{
  "status": "completed",
  "starting_checkpoint": "lesson37_best.npz",
  "starting_sha256": "9d306c2bc19218748c19ccf96d2f97706f16c2e7180fa9013007661b732e97c6",
  "train_sha256": "0cee762b4832c3a1c37c84c245ca912c51a1c22fa8df4c37e811ec735e390a69",
  "config": {
    "epoch": 5,
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "batch_size": 16,
    "seq_length": 32,
    "max_norm": 1.0,
    "chapter_shuffle_seed": 46
  },
  "history": [
    {
      "validation_loss": 5.937874743698944,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，只是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风一个了。”如此了，我见也去了。”我人得有两个事？”那者道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人",
          "repeat4": 0.8771929824561404
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“我日有个好去。”一个家：“这个里，见个了人大家。”一个道：“哥里，一个人不是道：“不个那，便了道：“你了的子。”我是道",
          "repeat4": 0.01754385964912286
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了。”那个道：“你是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我人，不是我",
          "repeat4": 0.7719298245614035
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了，一个在个大上。”这个要上人。”说人见一个来头？”那人道：“那好人人，不是道：“我要说：“老里，这个人人了。”贾见道：“",
          "repeat4": 0.0
        }
      ],
      "local_step": 0,
      "global_step": 19148,
      "adam_steps": 0,
      "checkpoint": "lesson37_best.npz"
    },
    {
      "validation_loss": 5.980868139271782,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，只不道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道",
          "repeat4": 0.7017543859649122
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风之。我了一回，可道：“便将的不得，又然一里，即到说：“我是有下尚，便是一个一个不知。\n者兵大得，何道：“者人之着，如了一",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得",
          "repeat4": 0.7543859649122807
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“你去。”正明道：“这又笑道：“不是大子，又叫他得，都道：“宝过，便道：“你见老了道：“众中，一道：“我在僧将之、，不是",
          "repeat4": 0.0
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了，只不得道：“我不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一个一个，不得道：“我不得，一",
          "repeat4": 0.7017543859649122
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了。”二明道：“者道：“小知马道：“我下的子，到此马，便是一个的都道：“我不玉道：“是又，把上不得道：“我宝，是上人一个，",
          "repeat4": 0.03508771929824561
        }
      ],
      "local_step": 200,
      "global_step": 19348,
      "adam_steps": 200,
      "checkpoint": "lesson41_step0200.npz"
    },
    {
      "validation_loss": 5.8086367838675175,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，不是我来。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢",
          "repeat4": 0.5964912280701755
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风来，与大人引、者曰：“刘逵。”操人道：“大日下子。”公曰：“今又不敢了。”玄曰：“他自见这般人，自不知我来，将于来来，不",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“我不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不",
          "repeat4": 0.6491228070175439
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“吾也说：“大来，以然那下山下，都是二来，你看这将人来，不见此而，也不知：“你曰：“今不何了。”三曰：“何然今了，便不敢",
          "repeat4": 0.0
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢不敢。”李曰：“吾曰：“吾不知我，不敢不敢。”李",
          "repeat4": 0.6842105263157895
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了，我如吾不我一””王曰：“曰：“此人如有了，再于去了。”操德之到，却有一个军来，你日也不日。”行者道：“今将三明曰：“吾",
          "repeat4": 0.0
        }
      ],
      "local_step": 1000,
      "global_step": 20148,
      "adam_steps": 1000,
      "checkpoint": "lesson41_step1000.npz"
    },
    {
      "validation_loss": 5.366861238436288,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人",
          "repeat4": 0.8596491228070176
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风上，叫了一点。众人一声。我那里来是个女子，便都一十一十余人，那面三个小子，把一件人有人，将那里去。”行者道：“他的不是，",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“你们是个人，不敢不得。”那里有一个人，不敢不得。”那里有一个人，不敢不得。”那里有一个人，不敢不得。”那里有一个人，不",
          "repeat4": 0.7017543859649122
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“这话子，可可去了。”玄德见一个小人、二爷、个五五十十余人，见手中一面，只见一条小儿，只是大圣，你们不是个家的子，却是一",
          "repeat4": 0.0
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了，只见那里有一个，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见一个人来，只见",
          "repeat4": 0.736842105263158
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了，你倒有一位，来了一个时，自己不曾，这个头头头、解、一个，径起军来，把一个十个十十百百人，那里去来。”行者道：“这里是个",
          "repeat4": 0.0
        }
      ],
      "local_step": 3000,
      "global_step": 22148,
      "adam_steps": 3000,
      "checkpoint": "lesson41_step3000.npz"
    },
    {
      "validation_loss": 5.118321193317643,
      "validation_positions": 316960,
      "samples": [
        {
          "prompt": "春风",
          "mode": "greedy",
          "seed": 42,
          "text": "春风，不知道：“你这个是个人，不知是了，只见那个人都是一个，不得一个，不得一个，不知道：“你们也不知，不知是了，我们不知道：",
          "repeat4": 0.24561403508771928
        },
        {
          "prompt": "春风",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "春风。”晁曰：“主大圣，一句，何不敢为我！”遂令吕军等一面，便回入京，那里有一个人马，说了一条。\n却说：“不见他的话，今日不",
          "repeat4": 0.0
        },
        {
          "prompt": "唐僧道：",
          "mode": "greedy",
          "seed": 42,
          "text": "唐僧道：“你们也不知，不知是了，我们不知道：“你这个是个人，不知是了，我们不知道：“你这个是个人，不知是了，我们不知道：“你这个",
          "repeat4": 0.5614035087719298
        },
        {
          "prompt": "唐僧道：",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "唐僧道：“我一时，与你看了，你要吃。”行者道：“此言在贾母，也不知：“正是：“不知这一时，说是：\n话说：“不知，是何明”说的，不",
          "repeat4": 0.01754385964912286
        },
        {
          "prompt": "宝玉听了",
          "mode": "greedy",
          "seed": 42,
          "text": "宝玉听了，只见一个人，不知道：“你这个是个人，不知是了，只见那个人都是一个，不得一个，不得一个，不知道：“你们也不知，不知是了，",
          "repeat4": 0.26315789473684215
        },
        {
          "prompt": "宝玉听了",
          "mode": "topk40_t07",
          "seed": 42,
          "text": "宝玉听了，一齐是一头，乃是大圣大家的人，却不不曾要了你不在那里，只听得问道：“你们有个大圣，却说不见了，且说：“哥哥，有人都是一",
          "repeat4": 0.0
        }
      ],
      "local_step": 4787,
      "global_step": 23935,
      "adam_steps": 4787,
      "checkpoint": "lesson41_step4787.npz"
    }
  ],
  "completed_local_steps": 4787,
  "best_local_step": 4787,
  "best_validation_loss": 5.118321193317643,
  "best_checkpoint": "lesson41_best.npz",
  "optimizer_state_saved": false,
  "completed_global_steps": 23935,
  "adam_steps": 4787,
  "seen_windows": 76584,
  "seen_positions": 2450688,
  "last_batch_size": 8,
  "book_windows": {
    "三国演义.txt": 14818,
    "西游记.txt": 18052,
    "水浒全传.txt": 22415,
    "红楼梦.txt": 21299
  },
  "training_process_loss": 5.454197187552217,
  "adam_state_bytes": 5757784,
  "run_seconds": 720.263348700013
}
```

老师对固定生成样例的观察：轮末比起点更多出现“遂令”“却说”“与你看了”等小说式短片段，但仍有长循环、断裂语句和跨书人物混杂（如唐僧开头续写出现贾母）。完整验证损失从5.937874743698944降至5.118321193317643，下降约0.819554；这是当前预设Adam全轮配置相对同一起点的改善，不能称为与完整SGD同预算对照，更不能据此宣布已能流畅写作。下一步判断应继续结合全部固定样例，不只挑一句可读片段。

### 学习者第四十一课实操验收

学习者提交完整运行输出和lesson41.json，状态completed。起点损失5.937874743698944，局部200步5.980868139271782、1000步5.8086367838675175、3000步5.366861238436288、4787步5.118321193317643，最佳局部4787步正确。相对起点平均交叉熵降低约0.819554，这不是准确率提高0.819554。

完整覆盖76584窗口、2450688个目标、尾批8行；局部更新4787次，模型累计23935次，Adam历史t=4787。四书窗口计数为三国14818、西游18052、水浒22415、红楼21299。训练过程按位置加权平均损失5.454197187552217；实际含验证保存总耗时544.3秒，约9.1分钟，区别于老师预跑约12分钟，不将两者当作纯训练吞吐比较。本课预算是预定4787次更新和五个完整验证候选，不是时间硬上限。

当前学习者未回答本课两道概念题，也未另给人工阅读评价，不提前记录为答对或已掌握。老师对最终六段样例的观察：比起点出现更多小说式短语，但贪心仍循环，top-k出现唐僧续写提到贾母、语句断裂等现象，尚未达到流畅续写。短200步未改善而整轮改善，只说明该配置这次轨迹如此，不保证所有短实验延长后都变好。

脚本语法树与完整预跑参考代码一致，仅清理行末空白并检查每个非空代码行中文注释。只读CI新增check_lesson41.py，独立重算五个候选完整验证，核对整轮覆盖、200步前缀参数及最佳文件五组参数一致，固定30段文字完整且与编号及重复统计对应，不重新训练一整轮。代码、报告、五份权重、模型说明、笔记及CI按约定完整上传。用户新增语料来源链接保留原内容并同步个人笔记，未改变本轮数据、词表或划分，也未将data/LightNovel5000_v1_txt目录上传；后续语料扩充需独立记录来源、去重和划分后再比较。

## 第四十二课：有预算上限的实际RNN/GRU对照

本课按用户最新授权由助手实现和代跑；学习者未提交该实现的手写代码或问答，不记为个人理解验收。完整方法、数学、参数量、环境和恢复说明见[lesson42_bounded_training.md](lesson42_bounded_training.md)，逐行注释代码见scripts/experiments/bounded_recurrent.py。

每组最多两整轮、2394更新，360秒批次边界截止（收尾验证保存另计），同数据、同Adam、同公共参数初始化。两组均实际完成2394步、4901376个目标，训练数据顺序指纹一致。新RNN最佳4.774707、GRU最佳5.200225，旧NumPy参考5.118321，因此当前保留新RNN，而不是因为GRU名称就替换。较旧模型改变的还有宽度、层数和窗口打乱等条件，不能将新旧差异只归因于某一种结构或GPU。

固定文字已有可理解短语但仍重复和不连贯。两组全部原始文字保存在lesson42.json，未将低重复率当作流畅性结论。完整检查点保存模型、Adam、随机状态、数据排列及游标，实际GPU下一批恢复核验差为0，CPU再核验。旧脚本和旧权重均保留。

另外从固定版本joe-chiu/TinyChineseStories采集50条格式候选，最多读取1MiB；样本76981字节，保存来源、许可声明、原始索引和SHA-256。它没有参与本次四书对照，也未建立任何新的训练/验证划分。原本地LightNovel5000大目录不上传、不混入当前数据。

学习者异步确认选择PyTorch自动求导，同时保留逐行中文注释和数学讲解。新增独立generate_recurrent.py用于加载本次最佳模型续写，不启动训练；固定种子和top-k=1退化行为纳入CPU核验。当前已代跑实验与此选择一致，不记录为学习者已手写通过。

## 第四十三课：保留Adam历史继续训练

按用户授权由助手实现并代跑，学习者未提交手写代码或问答。固定额外两轮/2394次/360秒批次边界截止，保持数据、结构和学习率。源PyTorch RNN从累计2394继续至4788次，完整验证损失4.774707→4.588138，准确率22.17%→23.88%。实际增加4901376个目标，保留Adam动量、二阶矩和累计t；下一批恢复核验参数差为0。固定生成仍有循环和语意断裂，不记为流畅写作。详细数学、全部固定样例及练习见[lesson43_continued_training.md](lesson43_continued_training.md)。未改变测试集用途，也未混入新语料。

## 第四十四课：增加轮数与步数

学习者要求扩大训练预算，助手实际代跑新增8轮、9576更新，上限600秒批次边界截止，逐轮完整验证；保持结构、数据、Adam历史和学习率。实际阶段耗时145.47秒，累计PyTorch训练12轮、14364步。完整验证损失4.588138→4.444933，准确率23.88%→25.43%。各轮均改善但增益减小，不能只凭此判定过拟合。采样局部更可读，仍有跨书人物混杂和贪心长循环。下一批恢复参数差0，Adam步数14364，最佳和最后均在最终位置。完整数学、命令和六段最终样例见[lesson44_more_epochs.md](lesson44_more_epochs.md)，各轮54段原始文字见lesson44.json。未记录为学习者已手写掌握。

## 第四十五课：新语料与小型Transformer

按学习者要求，助手代采集并实现、实际训练。固定版本TinyChineseStories共2277篇，按主题260组成组划分，训练1801/验证231/测试245篇；名字与类型变体留在同组，规范化完全重复0条，近似重复尚未检查。与原四书训练章节合并后3202702正文字符，只从训练集建立6799符号的新词表，保留BOS/EOS与尾段掩码。原四书与RNN产物保留。

4层4头D128、上下文128、共享输入输出权重，共1680000参数；实际6轮4896次更新、19229226目标、阶段93.35秒。最终混合完整验证4.132147，四书4.320604，新故事3.492973，均好于各自字频基线。与旧RNN不是公平结构对照。固定生成仍重复、人物混杂、性别错乱，未宣布流畅写作。因果遮罩、完整梯度、数据分组、词表来源、尾段覆盖、预算与CPU完整验证均检查；不记录为学习者已手写掌握。详细分块方法、公式、参数量和FLOPs见[lesson45_tiny_transformer.md](lesson45_tiny_transformer.md)。

第四十五课首次GitHub检查定位到Windows清单CRLF哈希与Linux检出LF的差异。修正为兼容同一内容的两种换行序列，并使后续训练按LF计算清单指纹；原始训练报告、语料、模型参数和指标均未修改。

## 第四十五课问答复盘与聊天目标更新

学习者连续完成注意力、残差、输出形状、生成索引、teacher forcing、损失加权、梯度、裁剪、step/epoch与断点续训问答。真实错答与纠正过程已逐项记录在[lesson45_qa_review.md](lesson45_qa_review.md)，不将答对小题等同于独立编写源码。拆头漏头维、训练/生成前文、backward/step、累计step和AdamW恢复是重点复习项；最后对两轮后下一步答1633正确。

学习者要求参考MiniMind实现小于1GB且能chat的模型。已核验固定提交2de2308d71750b44d56fd28cd8756239eee18957的当前主线结构、SFT数据掩码及训练入口；落实[chat_training_roadmap.md](chat_training_roadmap.md)和机器可读预算。首选30M方案FP32约120.10MB，64M候选约255.65MB，均为公式估算、未创建训练。当前168万参数续写模型未变成聊天模型，尚未运行大模型预训练或SFT。

新增下一课scripts/learn/26_chat_sft_format.py及两条原创格式示例，实跑验证完整多轮、角色标记、回答首字错位、仅助手正文与结束符参与损失；两例有效目标分别23与19。示例词表独立，不替换旧权重词表，不当作训练或评测证据。新增参数预算脚本，所有新增代码均逐行中文注释。

## 第四十六课：学习者对话格式实操验收

学习者提交26_chat_sft_format.py完整运行结果，两条对话输入长度39/36、有效回答目标23/19，助手复查源数据并复跑演示与独立检查，完全一致。第一条两个回答11+10字符和两个结束标记合计23，第二条18字符加结束标记合计19；无效目标位置分别16和17。格式检查通过，不表示已训练聊天模型，也不记为学习者已独立编写该实现。用户输入作为上下文、仅助手目标参与损失的概念尚待学习者回答确认。详细解释与下一步见[lesson46_chat_format.md](lesson46_chat_format.md)。

### 第四十六课：掩码问答与缩短回答实操验收

已补记用户问题保留、-100仅屏蔽直接损失、回答梯度仍经注意力影响问题相关参数等纠正；用户正文监督布尔列表、False拼写、首字与结束符、两轮teacher forcing的作答也已按真实表现记录。学习者将第一条示例末回答由10字符改成“好”，输出输入30/有效14，第二条仍36/19；助手检查实际文件并复跑演示与独立检查，均通过。第一条有效目标=(11+1)+(1+1)=14，合并有效分母33。JSON压缩空白不是内容错误，保留用户修改。此处“好”只作格式实验，不是“刚才我说了什么？”的合格语义答案，不用于真实训练或测评。详细过程与数据质量区别见[第四十六课笔记](lesson46_chat_format.md)。

## 第四十七课：分组划分与冻结版本（助手预备，待学习者实操）

已归档内容质量、重复、梯度依赖、组间泄漏、种子复现、增量组归属和验证尺子一致性问答，保留将验证组误答为训练集后纠正的记录。脚本27_split_chat_groups.py已由助手预跑两次，10组按8/1/1分配，12样本按8/3/1分配，正文字符100/81/11；g00苹果模板全在验证集。记录映射、输入与产物指纹，现有版本禁止因输入/种子变化或文件改动而静默覆盖。未执行新模型训练，未记录学习者已完成本课。

学习者问何时训练自己的小模型，已安排本课后两次实操推进正式语料/分词器及30M模型硬件20步试跑，检查通过再在当次启动最多1000更新/1800秒首段预训练；这是启动训练计划，不是聊天效果承诺，数据或环境检查失败需先修正。详见[第四十七课](lesson47_group_split.md)。

### 第四十七课学习者运行验收（2026-11-24）

学习者提交连续两次运行输出，均为train 8组/8样本/100字符、validation 1组/3样本/81字符、test 1组/1样本/11字符。助手核对本课相关文件无新增修改，复跑脚本和独立检查均通过，实际映射g00=validation、g01=test，其余八组=train。已完成现成脚本的运行与复现验收，不记为独立手写完成，不代表正式语料或新聊天模型训练已完成。下一实操按既定安排接入正式语料与分词器。详细证据见[第四十七课验收](lesson47_group_split.md)。
