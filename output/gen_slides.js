// 論文進度說明投影片（高壓室模板）
const PptxGenJS = require('pptxgenjs');
const util = require('/Users/ccli/.claude/skills/pptx-template/scripts/pptx_util');
util.setBG('/Users/ccli/Downloads/icml-claude/背景圖片.png');
const { bg, header, card, runs, BLUE, LBLUE, BORDER, CH, EN, SAFE } = util;

const OUT = process.argv[2] || '/Users/ccli/Downloads/effective-width-claude/output/論文進度說明_高壓室.pptx';
const FIG1 = '/Users/ccli/Downloads/effective-width-claude/results/figures/fig1_profile.png';

const pres = new PptxGenJS(); pres.layout = 'LAYOUT_16x9';
const RED = 'B03A2E', GREEN = '2E7D32', GREY = '666666', DARK = '1A1A1A';
let page = 0;
const newSlide = (title) => { const s = pres.addSlide(); bg(s); page += 1; if (title) header(pres, s, page, title); return s; };

// 條列：每個項目一個段落（breakLine 放在該項目最後一個 run 上，項目符號才會逐段出現）
function bullets(s, items, x, y, w, h, size = 14) {
  const tr = [];
  items.forEach((item, i) => {
    const rs = runs(item, { bullet: { indent: 12 } });
    if (i < items.length - 1) rs[rs.length - 1].options.breakLine = true;
    tr.push(...rs);
  });
  s.addText(tr, { x, y, w, h, fontSize: size, lineSpacingMultiple: 1.3, paraSpaceAfter: 4, align: 'left', valign: 'top', wrap: true, color: '333333' });
}
// 大數字 callout
function bigNum(s, x, y, w, num, label, color = BLUE) {
  s.addShape(pres.ShapeType.rect, { x, y, w, h: 1.25, fill: { color: LBLUE }, line: { color: BORDER, width: 0.75 }, rounding: true });
  s.addText(runs(num, { bold: true, color }), { x, y: y + 0.08, w, h: 0.62, fontSize: 26, align: 'center', valign: 'middle' });
  s.addText(runs(label, { color: '444444' }), { x: x + 0.1, y: y + 0.7, w: w - 0.2, h: 0.5, fontSize: 11.5, align: 'center', valign: 'top', wrap: true });
}
// 一行文字（混合字型）
function text(s, str, x, y, w, h, size = 14, opts = {}) {
  s.addText(runs(str, opts), { x, y, w, h, fontSize: size, color: opts.color || '333333', valign: opts.valign || 'top', align: opts.align || 'left', wrap: true, lineSpacingMultiple: 1.25 });
}
// 底部一句話結論
function takeaway(s, str, y = 4.45) {
  s.addShape(pres.ShapeType.rect, { x: SAFE.l, y, w: SAFE.w, h: 0.5, fill: { color: BLUE }, line: { color: BLUE, width: 0 }, rounding: true });
  s.addText(runs(str, { bold: true, color: 'FFFFFF' }), { x: SAFE.l + 0.15, y, w: SAFE.w - 0.3, h: 0.5, fontSize: 14, valign: 'middle', align: 'left' });
}

// ── 1 封面 ──
{
  const s = pres.addSlide(); bg(s); page += 1;
  s.addShape(pres.ShapeType.line, { x: 2.0, y: 1.45, w: 6.0, h: 0, line: { color: BLUE, width: 1.5 } });
  s.addText(runs('卷積神經網路的「寬度」到底用掉了多少？'), { x: 0.5, y: 1.6, w: 9.0, h: 0.8, fontSize: 32, bold: true, align: 'center', color: DARK });
  s.addText(runs('論文研究進度說明'), { x: 0.5, y: 2.4, w: 9.0, h: 0.5, fontSize: 20, align: 'center', color: '555555' });
  s.addText('How Much of a Convolutional Layer\'s Width Is Actually Used?', { x: 0.5, y: 2.9, w: 9.0, h: 0.4, fontSize: 14, align: 'center', color: '777777', fontFace: EN, italic: true });
  s.addShape(pres.ShapeType.line, { x: 2.0, y: 3.45, w: 6.0, h: 0, line: { color: BLUE, width: 1.5 } });
  s.addText(runs('報告人：李政崇　│　綜合研究所 高壓研究室　│　115 年 9 月'), { x: 0.5, y: 3.6, w: 9.0, h: 0.4, fontSize: 14, align: 'center', color: '777777' });
  s.addText(runs('投稿目標：IEEE 期刊（不投會議）'), { x: 0.5, y: 4.0, w: 9.0, h: 0.4, fontSize: 12, align: 'center', color: '999999' });
}

// ── 2 大綱 ──
{
  const s = newSlide('大綱');
  const items = ['這篇論文在問什麼？（一句話與比喻）', '為什麼值得做：跟台電的 AI 應用有什麼關係', '我們怎麼量、量了多少', '第一輪的三個發現', '誠實的自我審查：哪些說法站不住', '文獻查證與論文重新定位', '下一步、成本與時程'];
  items.forEach((item, i) => {
    const y = 1.12 + i * 0.5;
    s.addShape(pres.ShapeType.ellipse, { x: 0.5, y: y + 0.09, w: 0.3, h: 0.3, fill: { color: BLUE }, line: { color: BLUE, width: 0 } });
    s.addText(String(i + 1), { x: 0.5, y: y + 0.08, w: 0.3, h: 0.32, fontSize: 12, bold: true, align: 'center', color: 'FFFFFF', fontFace: EN });
    s.addText(runs(item), { x: 0.95, y, w: 8.5, h: 0.48, fontSize: 17, color: '222222', valign: 'middle' });
    if (i < items.length - 1) s.addShape(pres.ShapeType.line, { x: 0.95, y: y + 0.49, w: 8.0, h: 0, line: { color: 'E8E8E8', width: 0.5 } });
  });
}

// ── 3 一句話 ──
{
  const s = newSlide('這篇論文在問什麼？');
  text(s, '一個影像辨識的神經網路，每一層都有固定數量的「通道」——可以想成一條公路的車道數。設計者通常直接抄現成架構，或用大量算力搜尋出來。', 0.5, 1.1, 9.0, 0.9, 15);
  // 公路比喻：8 車道，實際用 4 條
  const lx = 0.9, ly = 2.0, lw = 8.2, laneH = 0.19;
  s.addShape(pres.ShapeType.rect, { x: lx, y: ly, w: lw, h: laneH * 8 + 0.1, fill: { color: '3C3C3C' }, line: { color: '3C3C3C', width: 0 } });
  for (let i = 0; i < 8; i++) {
    const y = ly + 0.05 + i * laneH;
    if (i > 0) s.addShape(pres.ShapeType.line, { x: lx + 0.1, y, w: lw - 0.2, h: 0, line: { color: 'FFFFFF', width: 0.75, dashType: 'dash' } });
    const used = i < 4;
    if (used) for (let k = 0; k < 6; k++) s.addShape(pres.ShapeType.roundRect, { x: lx + 0.35 + k * 1.3 + (i % 2) * 0.5, y: y + 0.035, w: 0.42, h: laneH - 0.07, fill: { color: i % 2 ? 'F4B942' : '5DA9E9' }, line: { color: '222222', width: 0.25 }, rectRadius: 0.03 });
  }
  s.addText(runs('名目寬度 C = 8 車道', { color: 'FFFFFF', bold: true }), { x: lx + lw - 2.2, y: ly + laneH * 4 + 0.15, w: 2.1, h: 0.6, fontSize: 12, align: 'right' });
  text(s, '我們逐層量的是：實際「在跑車」的車道有幾條（有效維度 k*），再除以總車道數 C。這個比例就是「寬度使用率」k*/C。', 0.5, 3.65, 9.0, 0.75, 13.5, { bold: true, color: BLUE });
  takeaway(s, '一句話：把「每一層用掉多少寬度」量準，是日後模型瘦身與設計寬度的前提。', 4.48);
}

// ── 4 為什麼值得做 ──
{
  const s = newSlide('為什麼值得做：跟台電有什麼關係');
  card(pres, s, 0.5, 1.1, 2.9, 1.6, '寬度 = 算力 = 成本', 'AI 模型要進現場設備（監測箱、邊緣裝置），每一層的通道數直接決定記憶體與運算量。用不到的寬度是白付的錢。');
  card(pres, s, 3.55, 1.1, 2.9, 1.6, '我們自己就在用卷積網路', '局部放電（PD）圖譜辨識、設備影像巡檢都是同一類模型。「該給多寬」目前靠經驗與試誤。');
  card(pres, s, 6.6, 1.1, 2.9, 1.6, '瘦身之前要先會量', '模型剪枝（pruning）現有方法都要先「給定」目標壓縮比；若能量出每層真正用掉多少，壓縮比就有依據。');
  text(s, '這篇論文不做新模型，做的是「量尺」：', 0.5, 2.85, 9.0, 0.4, 15, { bold: true });
  bullets(s, [
    '把兩篇文獻互相矛盾的量測放到同一批模型、同一批層上重做，找出矛盾來自哪裡',
    '補上文獻沒量過的架構（ResNet、ConvNeXt）與沒做過的對照（未訓練的同架構）',
    '找出三個會讓量測失真的陷阱，並提出修正——這是投稿的主軸',
  ], 0.5, 3.22, 9.0, 1.2, 13);
  takeaway(s, '定位：方法論／量測協定的論文，成果可直接回饋到模型瘦身與邊緣部署。', 4.45);
}

// ── 5 背景：寬度現在怎麼決定 ──
{
  const s = newSlide('背景：現在的寬度是怎麼決定的');
  const rows = [
    ['做法', '怎麼做', '問題'],
    ['抄現成架構', '直接沿用 VGG／ResNet 發表時的通道數', '那是別人為別的任務調的，未必適合自己的問題'],
    ['自動搜尋（NAS）', '給定算力預算，讓程式搜出一組寬度', '成本可達上千 GPU-hour；答案是預算的函數，不是問題的函數'],
    ['先做大再剪枝', '訓練大模型，再依「目標壓縮比」剪掉通道', '壓縮比要人先給；剪多少才對，方法本身不知道'],
  ];
  const tbl = rows.map((r, i) => r.map(c => ({ text: runs(c, i === 0 ? { bold: true, color: 'FFFFFF' } : {}), options: { fill: { color: i === 0 ? BLUE : (i % 2 ? 'FFFFFF' : LBLUE) }, valign: 'middle' } })));
  s.addTable(tbl, { x: 0.5, y: 1.15, w: 9.0, colW: [1.9, 3.3, 3.8], rowH: [0.42, 0.62, 0.62, 0.62], fontSize: 12.5, border: { type: 'solid', color: BORDER, pt: 0.75 }, align: 'left' });
  text(s, '共同點：沒有一個方法回答「這一層實際上用了多少寬度」。文獻裡問過這個問題的只有兩篇，而且答案互相矛盾。', 0.5, 3.6, 9.0, 0.7, 14, { bold: true, color: BLUE });
  takeaway(s, '先描述（量得準），再處方（該給多寬）。這篇做前者。', 4.45);
}

// ── 6 文獻矛盾 ──
{
  const s = newSlide('文獻的矛盾：同一個問題，兩種答案');
  const chartOpts = (title, color) => ({
    x: 0, y: 0, w: 4.3, h: 2.35, showTitle: true, title, titleFontSize: 12, titleFontFace: CH, titleColor: '333333',
    chartColors: [color], lineSize: 2.5, lineDataSymbol: 'circle', lineDataSymbolSize: 5,
    showLegend: false, catAxisTitle: '層數（淺 → 深）', showCatAxisTitle: true, catAxisTitleFontSize: 10, catAxisTitleFontFace: CH,
    valAxisTitle: '寬度使用率', showValAxisTitle: true, valAxisTitleFontSize: 10, valAxisTitleFontFace: CH,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFontSize: 9, catAxisLabelFontSize: 9, valAxisMajorUnit: 0.5,
    catAxisLabelFontFace: EN, valAxisLabelFontFace: EN, plotArea: { fill: { color: 'FFFFFF' } },
  });
  const labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
  s.addChart(pres.ChartType.line, [{ name: 'Garg 2019', labels, values: [0.2, 0.45, 0.7, 0.85, 0.9, 0.9, 0.85, 0.6, 0.3, 0.15] }],
    { ...chartOpts('Garg 等人 2019：先升後崩（駝峰）', RED), x: 0.5, y: 1.1 });
  s.addChart(pres.ChartType.line, [{ name: 'Elmoznino 2024', labels, values: [0.1, 0.15, 0.22, 0.28, 0.35, 0.42, 0.48, 0.55, 0.62, 0.7] }],
    { ...chartOpts('Elmoznino & Bonner 2024：一路上升', GREEN), x: 5.2, y: 1.1 });
  text(s, '（示意圖，非實際數據）', 0.5, 3.42, 9.0, 0.3, 10, { color: '999999', align: 'center' });
  bullets(s, [
    '兩篇用的「維度」定義不同、估計方式不同（一個逐位置取樣、一個先整張圖平均）、資料集與類別數也不同',
    '沒有人把這些因素拆開來看——所以連「它們是否真的矛盾」都不知道',
  ], 0.5, 3.68, 9.0, 0.8, 12);
  takeaway(s, '我們的切入點：在同一批模型、同一批層上，把兩種算法一起算，看差異來自哪裡。', 4.5);
}

// ── 7 怎麼量 ──
{
  const s = newSlide('我們怎麼量：四個步驟');
  const steps = [
    ['1. 餵圖片', '把 ImageNet 驗證集 5 萬張圖片送進已訓練好的公開模型'],
    ['2. 記錄每一層', '在每個卷積層出口記下「激活值」——該層對每張圖、每個位置的輸出'],
    ['3. 算主成分', '對各層通道做主成分分析（PCA），看變異集中在幾個方向'],
    ['4. 除以寬度', '達到 95% 變異需要的成分數 k*，除以通道數 C，得到使用率 k*/C'],
  ];
  steps.forEach((st, i) => {
    const x = 0.5 + i * 2.28;
    s.addShape(pres.ShapeType.rect, { x, y: 1.15, w: 2.1, h: 1.9, fill: { color: LBLUE }, line: { color: BORDER, width: 0.75 }, rounding: true });
    s.addText(runs(st[0], { bold: true, color: BLUE }), { x: x + 0.1, y: 1.22, w: 1.9, h: 0.4, fontSize: 14 });
    s.addText(runs(st[1]), { x: x + 0.1, y: 1.62, w: 1.9, h: 1.4, fontSize: 11.5, color: '444444', valign: 'top', wrap: true });
    if (i < 3) s.addText('▶', { x: x + 2.08, y: 1.85, w: 0.25, h: 0.4, fontSize: 12, color: BLUE, align: 'center', fontFace: EN });
  });
  text(s, '兩個對照，文獻沒做過：', 0.5, 3.25, 9.0, 0.35, 14, { bold: true });
  bullets(s, [
    '同一個架構、未訓練（隨機初始化）的版本也量一次：分開「學到的結構」和「架構本身就有的結構」',
    '每一層同時用兩種估計方式（逐位置取樣 vs 整張圖先平均）：直接檢驗文獻矛盾是不是估計方式造成的',
    '樣本數紀律：樣本數不到通道數的 50 倍的層一律不報告——樣本太少會憑空造出「低維」假象',
  ], 0.5, 3.6, 9.0, 1.35, 12.5);
}

// ── 8 規模與成本 ──
{
  const s = newSlide('第一輪量測：規模與成本');
  bigNum(s, 0.5, 1.15, 2.1, '4 + 2', '種架構（VGG-16、ResNet-18/50、ConvNeXt-T）＋ 2 個未訓練對照');
  bigNum(s, 2.8, 1.15, 2.1, '108 層', '通過樣本數門檻的卷積層，每層都有完整的特徵值頻譜');
  bigNum(s, 5.1, 1.15, 2.1, '50,000 張', 'ImageNet 驗證集全部圖片，每層每張圖取 16 個位置');
  bigNum(s, 7.4, 1.15, 2.1, 'US$ 0.90', '租用雲端 GPU 一趟的總花費（不購置硬體）', GREEN);
  bullets(s, [
    '量測管線自己寫，含單元測試；每一個數字都可由腳本重算（可重現性是投稿的基本要求）',
    '分析計畫在看到資料之前就先寫好（預先登記），之後所有偏離都留紀錄——避免「看到結果才挑說法」',
    '原始特徵值全部存檔，日後換指標不必重跑',
  ], 0.5, 2.7, 9.0, 1.4, 13);
  takeaway(s, '量測本身近乎免費；成本在分析與寫作，不在算力。', 4.45);
}

// ── 9 第一輪結果圖 ──
{
  const s = newSlide('第一輪結果：四種架構的逐層使用率');
  s.addImage({ path: FIG1, x: 0.5, y: 1.08, w: 4.55, h: 3.85 });
  text(s, '橫軸：層的相對深度（0 淺 → 1 深）\n縱軸：寬度使用率 k*/C\n藍：訓練後　紅：未訓練（隨機初始化）', 5.3, 1.1, 4.2, 0.95, 11.5, { color: GREY });
  bullets(s, [
    '沒有一個架構接近 1.0：訓練後的網路普遍只用到名目寬度的四到六成',
    'ResNet-50 特別低、而且鋸齒狀劇烈——這後來被證明是量法的問題，不是網路的問題（下一頁）',
    '訓練後（藍）與未訓練（紅）的曲線可以分得開：訓練確實留下了可量的痕跡',
    '曲線的「形狀」（駝峰或上升）在第一輪被過度解讀——後面會誠實說明',
  ], 5.3, 2.15, 4.2, 2.8, 12);
}

// ── 10 發現一：分母 ──
{
  const s = newSlide('發現一：分母的陷阱');
  const widths = ['64', '128', '256', '512', '1024', '2048'];
  s.addChart(pres.ChartType.bar, [
    { name: '除以名目寬度 C', labels: widths, values: [0.23, 0.34, 0.40, 0.37, 0.11, 0.13] },
    { name: '除以可達秩 r_max', labels: widths, values: [0.23, 0.34, 0.42, 0.42, 0.43, 0.47] },
  ], {
    x: 0.5, y: 1.1, w: 4.6, h: 3.3, barDir: 'col', barGrouping: 'clustered', chartColors: [RED, BLUE],
    showLegend: true, legendPos: 'b', legendFontSize: 10, legendFontFace: CH,
    catAxisTitle: '該層的通道數 C', showCatAxisTitle: true, catAxisTitleFontSize: 10, catAxisTitleFontFace: CH,
    valAxisTitle: '使用率（中位數）', showValAxisTitle: true, valAxisTitleFontSize: 10, valAxisTitleFontFace: CH,
    valAxisMinVal: 0, valAxisMaxVal: 0.6, valAxisMajorUnit: 0.2, valAxisLabelFontSize: 9, catAxisLabelFontSize: 9, catAxisLabelFontFace: EN, valAxisLabelFontFace: EN,
    showValue: false, plotArea: { fill: { color: 'FFFFFF' } },
  });
  text(s, 'ResNet-50 看起來特別低，因為 20/53 層是「1×1 擴張層」：輸入只有 C/4 個通道，輸出卻有 C 個。', 5.3, 1.1, 4.2, 0.75, 12.5, { bold: true });
  bullets(s, [
    '線性代數上，這種層的輸出秩最多只能到輸入通道數——使用率上限就是 0.25，跟網路學到什麼無關',
    '拿名目寬度 C 當分母，等於責怪一條四線道公路「為什麼沒跑滿八線」',
    '改用該層「可達到的秩」r_max 當分母：最寬的兩群層從 0.11、0.13 回到 0.43、0.47，整條曲線變平',
    '受限層 vs 不受限層：用 C 比較 p = 1.4×10⁻⁹（看似兩群）；用 r_max 比較 p = 0.49（差異消失）',
  ], 5.3, 1.85, 4.2, 2.6, 11.5);
  takeaway(s, '「寬層只用一成寬度」是分母的假象，已撤回；修正後 ResNet-50 與 VGG、ResNet-18 同級。', 4.45);
}

// ── 11 發現二：池化 ──
{
  const s = newSlide('發現二：「先平均再量」會一致地把數字壓低');
  text(s, '文獻裡數字偏低的那一篇（Elmoznino & Bonner）是把整張圖的特徵先平均成一個點再量；我們兩種方式在同一層上並排算：', 0.5, 1.1, 9.0, 0.7, 13.5);
  bigNum(s, 0.5, 1.95, 2.7, '108 / 108', '層：先平均的估計值全部比逐位置取樣低，無一例外');
  bigNum(s, 3.65, 1.95, 2.7, '水準差', '平均把空間變異抹掉，整條曲線往下平移');
  bigNum(s, 6.8, 1.95, 2.7, '不是形狀差', '「平均改變了曲線形狀」四個架構都不顯著（p 0.24–0.87）', GREY);
  bullets(s, [
    '這解釋了為什麼那篇文獻的絕對數字偏低，但不能拿來解釋文獻之間「形狀」的矛盾——第一輪曾這樣寫，審查後改掉',
    '108/108 無一例外本身有說服力；但相鄰層彼此相關，所以論文不引用 p 值的位數，只報計數',
  ], 0.5, 3.35, 9.0, 1.0, 12.5);
  takeaway(s, '文獻比較必須先對齊「估計方式」，否則比的是估計方式，不是網路。', 4.45);
}

// ── 12 發現三：trained vs random ──
{
  const s = newSlide('發現三：訓練確實留下可量的痕跡');
  text(s, '同一個架構、未經訓練的版本，本身就有一條使用率曲線（來自架構與輸入統計）。訓練後的曲線與它不同嗎？', 0.5, 1.1, 9.0, 0.7, 13.5);
  const rows = [
    ['架構', '訓練後較高的層數', '配對檢定（Wilcoxon）'],
    ['ConvNeXt-T', '17 / 22', 'p = 6.3×10⁻⁴'],
    ['ResNet-50', '44 / 53', 'p = 4.2×10⁻⁶'],
  ];
  const tbl = rows.map((r, i) => r.map(c => ({ text: runs(c, i === 0 ? { bold: true, color: 'FFFFFF' } : {}), options: { fill: { color: i === 0 ? BLUE : (i % 2 ? 'FFFFFF' : LBLUE) }, align: 'center', valign: 'middle' } })));
  s.addTable(tbl, { x: 1.5, y: 1.9, w: 7.0, colW: [2.2, 2.4, 2.4], rowH: 0.45, fontSize: 13, border: { type: 'solid', color: BORDER, pt: 0.75 } });
  bullets(s, [
    '這是文獻沒做過的對照。若兩者分不開，整篇的框架就要改成「這是架構的性質，不是學習的性質」——結果是分得開',
    '但差異幅度不大（平均 +0.09 到 +0.11），ConvNeXt 的差異只有層間抖動的一半——論文會如實寫出',
  ], 0.5, 3.4, 9.0, 1.0, 12.5);
  takeaway(s, '三個發現都用「逐層配對檢定」、用全部通過門檻的層——這是審查後改用的正確做法。', 4.45);
}

// ── 13 自我審查 ──
{
  const s = newSlide('誠實的自我審查：哪些說法站不住');
  text(s, '在投稿前，用 IEEE 審稿人的標準對第一輪結果做了一次敵意審查。結論：量測管線乾淨，但「曲線形狀」的主張全部不成立。', 0.5, 1.1, 9.0, 0.7, 13.5);
  card(pres, s, 0.5, 1.9, 4.4, 1.2, '① 形狀檢定沒有檢力', 'VGG-16 只有 13 層，「最後三分之一」是 4 個點的相關係數（p = 0.167）。全篇 36 個這類檢定、零多重比較校正，無一通過。');
  card(pres, s, 5.1, 1.9, 4.4, 1.2, '② 宣稱的效果小於自身抖動', 'VGG-16 宣稱的「終端塌縮」0.059，比同一條曲線相鄰層的階差中位數 0.074 還小。');
  card(pres, s, 0.5, 3.2, 4.4, 1.2, '③「重現」是事後挑的', '與 Garg 對照前 8 層 r = 0.97，全部 13 層 r = 0.13；而且資料集不同。論文不再使用「重現」二字。');
  card(pres, s, 5.1, 3.2, 4.4, 1.2, '④ n = 1', '每個架構只有一組公開權重，沒有誤差棒。任何「形狀」主張在取得多組權重之前都不能寫。');
  takeaway(s, '寧可自己先否掉，也不要被審稿人否掉。倖存的三個發現改用正確的檢定後仍然成立。', 4.5);
}

// ── 14 文獻查證 ──
{
  const s = newSlide('文獻查證：「分母」這個觀察是新的嗎？');
  text(s, '審查提出的最大風險：發現一的秩上界在線性代數上很淺顯，可能早有人講過。今日查證結果：', 0.5, 1.1, 9.0, 0.6, 13.5);
  const rows = [
    ['文獻', '講了什麼', '跟我們的差別'],
    ['Han 等人, CVPR 2021（ReXNet）', '明講「輸出秩受輸入維度限制」，畫過 秩/C 對 C_in/C_out 的圖', '隨機權重、當設計準則用；沒有拿它當量測的分母'],
    ['Kim 等人, 2018（預印本）', '寫出一般卷積秩上界 min(C_in·k², C_out)', '為了壓縮模型；沒有量訓練後網路的逐層剖面'],
    ['Bhojanapalli 等人, ICML 2020', 'Transformer 注意力頭的同一種秩瓶頸', '不同架構，同一個觀念'],
    ['量測文獻（Ansuini 2019、Elmoznino 2024）', '量 block 輸出而非卷積輸出，上界對它不成立', '沒中招，但也沒有處理——這是我們可以講清楚的點'],
  ];
  const tbl = rows.map((r, i) => r.map(c => ({ text: runs(c, i === 0 ? { bold: true, color: 'FFFFFF' } : {}), options: { fill: { color: i === 0 ? BLUE : (i % 2 ? 'FFFFFF' : LBLUE) }, valign: 'middle' } })));
  s.addTable(tbl, { x: 0.5, y: 1.7, w: 9.0, colW: [2.6, 3.3, 3.1], rowH: [0.36, 0.55, 0.5, 0.42, 0.55], fontSize: 10.5, border: { type: 'solid', color: BORDER, pt: 0.75 }, align: 'left' });
  takeaway(s, '結論：上界本身不新，「拿它當量測分母、並在訓練後網路上證明假象消失」是我們的。論文措辭改為「文獻忽略了一個已知上界」。', 4.2);
}

// ── 15 論文重新定位 ──
{
  const s = newSlide('論文重新定位：從「曲線形狀」改為「量測協定」');
  text(s, '原本的標題、假設與結果章節都繞著曲線形狀寫，現在改以三個經得起檢定的主張為主軸：', 0.5, 1.1, 9.0, 0.6, 13.5);
  const rows = [
    ['承重主張', '證據', '狀態'],
    ['① 分母要用可達秩 r_max，不是名目寬度 C', 'p = 1.4×10⁻⁹ → 0.49；受限層達上界的 93%', '強烈倖存（上界已知，用法是我們的）'],
    ['② 全域平均池化一致壓低水準', '108 / 108 層，無例外', '強烈倖存（新穎性待查，下週）'],
    ['③ 訓練後 ≠ 隨機初始化', 'Wilcoxon p ~ 10⁻⁴ – 10⁻⁶', '倖存'],
    ['④ 卷積輸出 vs block 輸出不可比', '今日查證直接掉出來的；待補一張對照', '新增，待量測'],
    ['曲線的形狀（駝峰／上升）', '36 個檢定無一通過校正', '不倖存，從論文移除'],
  ];
  const tbl = rows.map((r, i) => r.map((c, j) => {
    const isLast = i === rows.length - 1;
    const col = i === 0 ? 'FFFFFF' : (isLast ? RED : (j === 2 && c.startsWith('強') ? GREEN : '333333'));
    return { text: runs(c, { bold: i === 0, color: col }), options: { fill: { color: i === 0 ? BLUE : (i % 2 ? 'FFFFFF' : LBLUE) }, valign: 'middle' } };
  }));
  s.addTable(tbl, { x: 0.5, y: 1.72, w: 9.0, colW: [3.4, 3.0, 2.6], rowH: [0.36, 0.44, 0.44, 0.4, 0.44, 0.4], fontSize: 11, border: { type: 'solid', color: BORDER, pt: 0.75 }, align: 'left' });
  takeaway(s, '新框架：「逐層有效寬度的三個量測陷阱，以及修正後的結果」——貢獻更小，但每一條都站得住。', 4.35);
}

// ── 16 下一步 ──
{
  const s = newSlide('下一步、成本與時程');
  const rows = [
    ['項目', '目的', '成本', '狀態'],
    ['量 16–20 個公開權重（timm）', '解決 n = 1；含 6 個不同配方的 ResNet-50，以及擴張比不同的架構（MobileNetV2 等）驗證分母修正', '< US$3，約 1 GPU-hour', '清單與程式已備妥，待跑'],
    ['驗證關卡：以 CIFAR-10 對照 Garg 的公開結果', '管線至今沒有對任何已發表結果通過驗證', '約 US$4', '腳本已寫，未跑'],
    ['補「卷積輸出 vs block 輸出」對照', '對應主張 ④', '併入上一趟', '程式已加'],
    ['池化主張的文獻新穎性查證', '避免主張 ② 重演今天的事', '0（案頭工作）', '下週'],
    ['改寫 §I／§II 與標題、補摘要與結論', '依新框架重寫', '0', '取得多組權重的分佈之後'],
  ];
  const tbl = rows.map((r, i) => r.map(c => ({ text: runs(c, i === 0 ? { bold: true, color: 'FFFFFF' } : {}), options: { fill: { color: i === 0 ? BLUE : (i % 2 ? 'FFFFFF' : LBLUE) }, valign: 'middle' } })));
  s.addTable(tbl, { x: 0.5, y: 1.1, w: 9.0, colW: [2.5, 3.5, 1.5, 1.5], rowH: [0.36, 0.62, 0.5, 0.4, 0.4, 0.42], fontSize: 10.5, border: { type: 'solid', color: BORDER, pt: 0.75 }, align: 'left' });
  takeaway(s, '總算力花費預估 < US$10。作者獨立作業，不需研究經費、不購置硬體，只租雲端 GPU。', 4.2);
}

// ── 17 投稿與風險 ──
{
  const s = newSlide('投稿規劃與風險');
  card(pres, s, 0.5, 1.1, 4.4, 1.4, '投稿目標', 'IEEE 中上程度期刊（目前草稿以 IEEE TNNLS 格式撰寫），不投會議（不便出國）。期刊審稿週期長，以「主張少但每條站得住」為策略。');
  card(pres, s, 5.1, 1.1, 4.4, 1.4, '目前完成度', '量測管線與測試完成；論文 §I–V 有草稿；審查與文獻查證完成；引用已補。待：多組權重量測、驗證關卡、依新框架重寫。');
  text(s, '風險與因應：', 0.5, 2.6, 9.0, 0.35, 14, { bold: true });
  bullets(s, [
    '審稿人認為貢獻太小（上界已知）→ 主軸改為「量測文獻全忽略了它、修正後假象消失」，明確引用先行文獻',
    '池化主張也有先行文獻 → 下週查證；即使有，108/108 的配對比較仍沒人做過',
    '多組權重量出來後主張變弱 → 如實報告分佈；這正是先量再投的原因',
    '管線對 Garg 的驗證不通過 → 停下來修管線，全部重跑（成本仍 < US$5）',
  ], 0.5, 2.95, 9.0, 1.5, 11.5);
  takeaway(s, '原則：不寫任何一句撐不過自己敵意審查的話。', 4.5);
}

// ── 18 結語 ──
{
  const s = newSlide('結語');
  bullets(s, [
    '問題：卷積網路每一層的寬度到底用掉多少？這是模型瘦身與邊緣部署前該先量準的量',
    '做了：自建量測管線，在 4 種架構、108 層、5 萬張圖上完成第一輪，花費不到 1 美元',
    '發現：三個會讓量測失真的陷阱——分母、池化、量測點——並各自給出修正與證據',
    '誠實：第一輪對「曲線形狀」的解讀經自我審查後撤回；論文改以量測協定為主軸',
    '下一步：以不到 10 美元的算力補齊多組權重與驗證關卡，再依新框架改寫投稿',
  ], 0.5, 1.15, 9.0, 3.0, 14.5);
  takeaway(s, '謝謝指教。', 4.45);
}

pres.writeFile({ fileName: OUT }).then(() => console.log('wrote', OUT));
