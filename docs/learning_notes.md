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
