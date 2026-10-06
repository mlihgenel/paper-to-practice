# PaliGemma 2 deney sonuçları (Colab T4, 2026-10-01)
Ham sonuçlar: bu klasördeki `qa.csv`, `ocr.csv`, `fis.csv`, `altyazi.csv` (Google Colab, T4 GPU).

## Tabela QA (54 soru) — doğruluk % [katı / esnek]
| model | EN soru | TR soru |
|---|---|---|
| 224 | 22.2 / 33.3 | 31.5 / 40.7 |
| 448 | 37.0 / 63.0 | 46.3 / 66.7 |

Yazı boyu (katı %): büyük 224: EN 32.3, TR 45.2 · 448: EN 41.9, TR 51.6 | küçük 224: EN 8.7, TR 13.0 · 448: EN 30.4, TR 39.1
Soru tipi (katı %): sayı 224: 26.7/33.3 → 448: 66.7/66.7 | metin 224: 20.5/30.8 → 448: 25.6/38.5
Süre: 224 0.91 sn/soru, 448 3.08 sn | GPU tepe bellek 6.1 / 6.3 GB

## OCR (11 tabela)
| model | CER | CER esnek | TR harf doğru | katlanmış | diğer | TR doğru % |
|---|---|---|---|---|---|---|
| 224 | 13.8 | 11.9 | 77 | 8 | 20 | 73.3 |
| 448 | 19.7 | 18.9 | 99 | 5 | 1 | 94.3 |
Not: CER sıraya duyarlı. 448 c09'u farklı sırayla okudu (CER 1.06). c09 hariç: 224 %14.2, 448 %11.1.
224 uydurmaları: c03 "T.C. KIZILAY BELEDİYESİ" (doğrusu AYDIN BÜYÜKŞEHİR), c30 "20 MAYIS 1938 SALİH SABAHİ" (doğrusu 29 MAYIS 1453 SALI SABAHI), c26 "bakanın ürünleri" (doğrusu tütün), c23 "RAYAT EKMEK AŞKISI". 448 c30 kitabeyi neredeyse kusursuz okudu (CER 0.027).

## Fiş (CORD, 50 fiş) — toplam tutar doğru %
| model | EN | TR |
|---|---|---|
| 224 | 42 | 46 |
| 448 | 84 | 84 |

## 224 ile 448'in ayrıştığı sorular (TR)
448 doğru, 224 yanlış: camlik→camiatik, parar→pazar, "vali muammer gey", "balaban emek", kanun "2"→4207, somon "3,40"→8, Londra "895 km"→2502, Moskova "1437"→1757, hamam "1650"→1741, pano "kurucu askeri anıt"→kurtuluş yolu
224 doğru, 448 yanlış: sigorta şirketi (448 "balaban" dedi), c24 448 "rayli yurume" (harf kaybı)

## Altyazı (nitel)
"caption tr" bazen İngilizceye kayıyor: "lion sta(tue)", "golden", "In this image we can see a sign board..." (tr isteminde).

---
# Ham CSV analizi

## EN soru neden geride?
1. Çeviri: EN soruda model cevabı çeviriyor: "sigara" yerine "smoking", "alabalık" yerine "alabama" (224).
2. Pes etme: "unanswerable" sayısı 224: TR 0 / EN 2, 448: TR 2 / EN 5. 448 EN "pazar" mahallesine "unanswerable" dedi, TR'de doğru buldu.
3. Harf: EN cevaplarda Türkçe harf neredeyse hiç kalmıyor (448 EN 1/15, TR 3/14).

## QA'da Türkçe harf kaybı
Model metni doğru bulduğunda (esnek=1) Türkçe harfleri de doğru yazma oranı: 224 TR 3/8, EN 0/6 · 448 TR 3/14, EN 1/15.
Örnekler: acil cikis, cinarcik, akcay makine, daghan, turku, resat nuri guntekin, bayat ekmek askisi.
Karşıtlık: `ocr` modunda 448 Türkçe harflerin %94'ünü doğru yazıyor. Yani harfleri görüyor, sorulara cevap verirken düşürüyor (VQA eğitim cevapları muhtemelen ASCII'ye normalize edilmiş; tahmin).

## OCR: sıradan bağımsız kelime F1 (CER yerine)
224: katı 73.1, esnek 83.1 · 448: katı 87.0, esnek 91.6
448 kayıp: c09 (61). Satırları tekrarlıyor (tekrar döngüsü), c20'de "brother" 3 kez.
Uydurma: 224 c26 olmayan cümle uydurdu ("Sigara içenler uyarıdan ötekileşirler", "TÜRKİYE CUMHURİYETİ BAKANLIĞI"). 448 de c26'ya "GAZİAN / TÜRKIYE CUMHURİYETİ" ekledi.

## Fiş hataları (EN, yanlışlar)
224 (29 yanlış): başka sayı 11, eksik/fazla hane 11 (ör. 230,000 → "230.00", binlik ayırıcıyı ondalık sanıyor), 1 rakam hatalı 7
448 (8 yanlış): 1 rakam hatalı 5 (377,859 → 377,659), başka sayı 3 (ödenen nakit/para üstü: 100,000, 72000)
448 doğru & 224 yanlış: 23 fiş; tersi: 2.

## Altyazı: TR ile EN
EN altyazılar görseldeki yazıyı okuyor ("hamsi 4 mtl", "beypazarı kurusu", "Şehir Merkezi"); TR altyazılar genel ya da yanlış:
c31 448 TR "Invertebrate food is for sale in a red bucket." (TR istendi, EN + yanlış), c46 448 TR "Ağzında mavi bir şey tutan bir kadın" (yanlış),
c23 224 TR "In this image we can see a sign board..." (tamamen İngilizce), c06 TR "bir lion statue"/"golden statue" karışık.

## Büyük/küçük harf testi (2026-10-06)
Soru: QA'daki harf kaybı, cevapların normalize edilmesinden mi geliyor?
- QA: 216 cevabın (2 model × 2 dil × 54) hiçbirinde büyük harf yok.
- OCR (448): büyük/küçük harf altın metinle birebir (c01 11/8, c02 9/0, c03 51/0, c30 150/0 büyük/küçük). Küçük harfli c26'da Türkçe harf 25/25 doğru, yani küçük Türkçe harf tanıma sorunu yok.
- Silme tam değil: 448 TR "marmaranın inisi" ı'yı koruyor.
Sonuç: QA çıktısı küçük harfe ve büyük ölçüde ASCII'ye normalize. VQA tarzı cevap normalizasyonuyla tutarlı (eğitim verisi görülmediği için kesin değil).

## TR–EN farkının dökümü (katı puan)
448: TR doğru & EN yanlış 5, tersi 0.
- q24 "sigara" → EN "smoking" (çeviri, okuma doğru)
- q34 "alabalık" → EN "alabalik", q40 → EN "askeri guvenlik bolgesi girilmez" (harf kaybı, esnek=1)
- q03 → EN "sokak", q06 → EN "unanswerable" (gerçek fark)
Esnek: TR 36/54, EN 34/54.
224: TR doğru & EN yanlış 6 (q20 "studio", q22 harf, q24 "smoking", q29 sayı "2", q34 "alabama", q40 "gvenlik"), tersi 1 (q19 TR "balaban emek").

