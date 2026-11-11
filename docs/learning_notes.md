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
