# マルハン前橋 20スロ・フロアマップ V2

可視化専用の単独CLIです。朝自動化、通知、予測、Guard、Formal、データ取得には接続しません。
必要環境はPython 3.10以降、Pillow、日本語フォント（Windows標準のMeiryoを自動検出）です。

```powershell
python machine_number/maruhan_maebashi_floor_map_generator.py --daily data/maruhan_maebashi/machine_number/ana_slo_20260910.csv --data-date 2026-09-10
```

`--daily`省略時は日付形式が一致する最新ファイルを選びます。最新ファイルが不正でも過去日へ代替しません。
`--output-dir`、`--layout`、`--font`を指定できます。標準出力先はリポジトリの`outputs/floor_map`です。
日付別PNG/JSONと`latest.png`/`latest.json`を出力します。latestを過去日に戻す操作は拒否します。

## 三層と正本

1. `maruhan_maebashi_floor_map_layout.csv`：514台の台番号→固定位置。
2. daily CSV：日付・台番号・機種名。機種名の空白・記号・表記を変更せずJOIN。
3. Pillow：物理マスターを変えず、描画専用の島移動・倍率でコンパクト化。画像生成AI、乱数、予測情報は使用しません。

元資料はDownloadsの1536×884版`maruhan島図.png`で、OneDriveの2750×1583版とも番号・配置を照合しました。
両元画像のパスとSHA-256、島名、円形島、切欠きの説明は隣接するlayout JSONに保存しています。
チャット生成の装飾版には台番号重複等があるため、物理位置の根拠にしません。
元画像に記載された古い機種名は採用しません。元画像自体は生成時に不要です。

### 座標と特殊位置

座標は1536×884基準図の台枠中心を転記した模式図ピクセルで、左上原点・右がx正・下がy正です。
実測距離ではありません。直線島は規則間隔に整えています。CSVは各台の座標を明示しており、生成時に台番号から位置を推測しません。
`side`はN/S（上/下）、W/E（左/右）、RING。`position_order`は各side内の表示順です。
`slot_index`は空き位置を含む物理枠順で、欠番枠を詰めず、同一機種のラベルを空き位置越しに結合しないために使います。

- C03/S：842と843の間（849の下）に空き位置。
- C08/N：1015と1014の間（1006の上）に空き位置。
- 1114は901の下、1113は900の下、1112は885の下。
- C05：797〜813の17台を環状配置。813→797の隣接も保持。
- 581〜620はマスターに含めず、V2では台枠を描かず、対象外であることを脚注に表示。

円形島は元の角度・円周順・相対座標を保持して描画します。CSV/JSON内のisland_idと物理座標はV1のままです。
V2の表示座標はgeneratorの`display_positions`だけで計算します。島順・side内の台順・向きは変えません。
将来の色オーバーレイはmachine_noで結合可能ですが、予測との結合は実装していません。

## 表示と再現性

全体版は4030×3610px。V1の巨大な下段一覧、G番号、内部島ID、出入口表記を表示しません。
台番号は26pxから34pxへ拡大。同じ機種の連続範囲ごとにdaily原文を島の脇へ直接表示します。
横島は上/下、縦島は左右の帯（横書きを90度回転）に配置。円形島は中央、複数機種なら円形島脇の引出線付きラベルです。
同一機種でも別side・空き位置越しには結合しません。色は範囲を見分ける補助であり、機種固有色ではありません。

機種名は原則2行まで、狭い単独台では最大3行まで折り返し、28～12pxで適合させます。
文字を削除・省略・略称化しません。最小サイズでも入らない名前は切らずに生成を停止します。
514台をスマートフォンの全景表示だけで読めるわけではありません。全景で位置を把握し、ピンチ拡大で台番号・長い機種名を確認してください。
文字境界、ラベル同士、ラベルと文字、台枠同士の重なりを検証します。

同一CSV bytes・layout bytes・フォント・PillowバージョンならPNGはbyte単位で再現可能です。
生成時刻はJSONにだけ記録し、PNG内容には混ぜません。フォントとPillowの版もmetadataに記録します。

## Fail-Closed

日次CSVとマスターの各514行・514一意台番号・欠損0・重複0・5スロ混入0・集合完全一致を必須とします。
機種名欠損、日付/ファイル名不一致、座標/順序不正、マスターSHA不一致も停止対象です。
入力検証と全描画・文字境界/重複チェックが完了するまで出力を作成・上書きしません。
失敗時は`FLOOR_MAP_BLOCKED`と終了コード1を返し、古いlatestを現在日付の画像として作り直しません。

出力は一時ファイルへ完成させてから各ファイルをatomic replaceし、latest JSONを最後の完了マーカーにします。
複数ファイル全体が一括トランザクションになるわけではありません。外部利用時はJSONの`data_date`と`png_sha256`を確認してください。

マスター改定時は元資料で位置を確認し、CSVとJSONの`layout_sha256`を一緒に更新してレビューします。
このCSVだけ`.gitattributes`の`text eol=lf`で作業ツリー・Git保存・checkoutをLFに固定し、JSONのSHA-256もLFのbytesに対して検証します。
daily側の不一致に合わせてマスターを自動修正する処理はありません。

## テスト

```powershell
python -m pytest tests/test_maruhan_maebashi_floor_map.py -q
python -m pytest tests -q
```

保守対象の全体suiteは`tests/`です。リポジトリ直下の過去のブラウザ試行スクリプト等は実行しません。

## Windows通常ユーザーでの最終PNG生成

Codex側の試験PNGは検証用です。人間確認用PNGは通常ユーザーPowerShellで生成してください。
既存Sandbox出力へのアクセス拒否を避ける場合は、通常ユーザーが新規作成する出力先を指定します。
ACL/所有権は変更しません。Morning Automationを実行する必要はありません。

```powershell
Set-Location -LiteralPath "C:\Users\user\Desktop\Documents\SlotAnalyzer"
python -B machine_number/maruhan_maebashi_floor_map_generator.py --output-dir outputs/floor_map_v2_user
```

このコマンドは実行時点の最新dailyを検証し、日付別PNG/JSONとlatest PNG/JSONを作成します。
出力ファイル名は従来どおりmaruhan_maebashi_floor_map_<date>.pngとmaruhan_maebashi_floor_map_latest.pngです。
人間確認では特殊空白2か所、台番号、単独台の長い機種名、縦ラベルの読みやすさを確認してください。
