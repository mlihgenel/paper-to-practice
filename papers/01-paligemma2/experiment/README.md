# Deney: Türkçe tabelalar ve fişlerde PaliGemma 2 (224 ile 448 piksel)

[![Colab'da aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlihgenel/yz50-research/blob/main/papers/01-paligemma2/experiment/paligemma2_experiment.ipynb)

Çalıştırılmış hali (tüm çıktılarıyla): [Google Colab](https://colab.research.google.com/drive/1wL1xt4rzYNZ8w3tCMfDH7slQWwOQ7ezJ)

Makale, metin okuyan görevlerin yüksek çözünürlükten kazandığını söylüyor. Bu deney aynı soruyu makalenin bakmadığı bir veride, Türkiye'de çekilmiş tabela ve etiket fotoğraflarında test ediyor. Kullanılan iki model `google/paligemma2-3b-mix-224` ve `google/paligemma2-3b-mix-448`. İkisi de 3B, aradaki tek fark görüntü çözünürlüğü (256 ve 1.024 görüntü tokenı).

> "Mix" modeller birçok görevin karışımıyla fine-tune edilmiş, kutudan çıktığı gibi kullanılabilen sürümler. Makaledeki gibi her göreve ayrı fine-tune yapılmadı, bu yüzden rakamlar makaledekilerle birebir kıyaslanamaz. Bakılan şey aynı model ailesi içindeki **yön**.

## Ne ölçüldü?

| # | Deney | Veri | Ölçüm |
| --- | --- | --- | --- |
| 1 | Türkçe soru mu, İngilizce soru mu? | 41 tabela/etiket fotoğrafı, 54 soru | Doğru cevap oranı |
| 2 | 224 mü, 448 mi? | Aynı set (büyük ve küçük yazı ayrı) + fişler | Doğruluk, süre, bellek |
| 3 | Türkçe karakterler (ç ğ ı i ö ş ü) | 11 tabela, `ocr` komutu | Kelime eşleşmesi (F1), Türkçe harf doğruluğu |
| 4 | Fişte toplam tutar | [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) testinden 50 fiş | Doğru cevap oranı |

Puanlama iki türlü yapıldı. **Katı** puanda Türkçe harfler birebir doğru olmalı ("türkü" ≠ "turku"). **Esnek** puanda harf farkları görmezden geliniyor.

## Sonuçlar

| | 224 px | 448 px |
| --- | --- | --- |
| Büyük yazı (31 soru, TR, katı) | %45 | %52 |
| Küçük yazı (23 soru, TR, katı) | %13 | **%39** |
| Sayılar (15 soru, TR) | %33 | **%67** |
| Fişte toplam tutar (50 fiş) | %46 | **%84** |
| OCR kelime F1 (11 tabela) | %73 | %87 |
| OCR'da Türkçe harf doğruluğu | %73 | %94 |
| Süre / soru (T4) | 0,9 sn | 3,1 sn |
| GPU bellek | 6,1 GB | 6,3 GB |

![224 ile 448 karşılaştırması](../figures/original/resolution-224-vs-448.png)

Öne çıkan gözlemler:

- **Uydurma:** 224 okuyamadığı yerde akıcı ama yanlış metin yazıyor. Örnekler: "29 Mayıs 1453" → "20 Mayıs 1938", "Aydın Büyükşehir" → "Kızılay Belediyesi", Londra'ya 2.502 km yerine 895 km. ([görsel](../figures/original/galata-ocr.png))
- **Türkçe soru, İngilizce soru:** 448'de katı puanla Türkçe soru %46, İngilizce soru %37. Ama Türkçenin kazandığı 5 sorudan 3'ü puanlamadan geliyor: 1'inde model cevabı çeviriyor ("sigara" → "smoking"), 2'sinde sadece harf kaybı var. Esnek puanla iki dil neredeyse eşit (36'ya 34 soru). İngilizce soruda model daha sık "cevaplanamaz" diyor (5'e 2).
- **Harf kaybı:** Model `ocr`'da Türkçe harfleri tutuyor, soru-cevapta düşürüyor ("acil cikis", "cinarcik"). Metni doğru bulduğu 14 cevabın sadece 3'ünde harfler doğru. Soru-cevaptaki 216 cevabın hiçbirinde büyük harf yok, `ocr` ise büyük/küçük harfi koruyor. Bu, soru-cevap çıktısının normalize edildiğine işaret ediyor. ([görsel](../figures/original/same-sign-two-prompts.png))
- **Fiş hataları:** 224 binlik ayracını ondalık sanıyor (230,000 → "230.00"). 448'in hataları çoğunlukla tek rakam.

Tüm sayılar ve örnekler: [`results/summary.md`](results/summary.md)

## Sınırlar

- 54 soruda her soru yaklaşık 2 puan ediyor, yani tabeladaki birkaç puanlık farklar kesin değil. Fiş sonucu ise sağlam: 448'in doğru, 224'ün yanlış bildiği 23 fiş var, tersi 2.
- Sorular ve doğru cevaplar elle yazıldı.
- 10B ve 28B ücretsiz GPU'ya sığmadığı için "beyin" tarafı denenmedi.

## Nasıl çalıştırılır?

1. Hugging Face'te [mix-224](https://huggingface.co/google/paligemma2-3b-mix-224) ve [mix-448](https://huggingface.co/google/paligemma2-3b-mix-448) sayfalarında lisansı kabul et.
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) üzerinden "Read" yetkili bir token oluştur.
3. Notebook'u Colab'da aç, *Çalışma zamanı → T4 GPU* seç. Token'ı 🔑 *Secrets* bölümüne `HF_TOKEN` adıyla ekle ve "Notebook access"i aç.
4. *Tümünü çalıştır*. Toplam ~25 dakika sürer, sonunda sonuçlar zip olarak iner.

Notebook görselleri bu repodan (`data/images/`) indirir. Ulaşılamazsa Wikimedia Commons'a geçer.

## Dosyalar

| Dosya | Ne |
| --- | --- |
| [`paligemma2_experiment.ipynb`](paligemma2_experiment.ipynb) | Deney notebook'u (veri seti içine gömülü) |
| [`data/signs.json`](data/signs.json) | 41 görsel, 54 soru, 11 OCR metni, doğru cevaplar |
| [`data/questions.py`](data/questions.py) | Aynı soruların okunabilir listesi |
| [`data/images/`](data/images/) | Tabela fotoğrafları |
| [`data/SOURCES.md`](data/SOURCES.md) | Her fotoğrafın yazarı ve lisansı |
| [`results/`](results/) | Ham sonuçlar (`qa.csv`, `ocr.csv`, `fis.csv`, `altyazi.csv`) ve [`summary.md`](results/summary.md) |

## Lisanslar

- Tabela fotoğrafları: Wikimedia Commons. Her biri kendi lisansıyla (CC BY-SA, CC BY, CC0, kamu malı), ayrıntılar [`data/SOURCES.md`](data/SOURCES.md) içinde.
- Fişler: CORD-v2, CC BY 4.0.
- Modeller: [Gemma kullanım koşulları](https://ai.google.dev/gemma/terms).
