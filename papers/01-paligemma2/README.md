# 01 · PaliGemma 2: A Family of Versatile VLMs for Transfer

| | |
| --- | --- |
| **Yazarlar** | Steiner, A., Pinto, A. S., Tschannen, M., et al. (Google DeepMind) |
| **Yayın** | Aralık 2024 · [arXiv:2412.03555](https://arxiv.org/abs/2412.03555) · CC BY 4.0 |
| **Konu** | Görsel-dil modelleri (VLM), çözünürlük ve model boyutu, fine-tuning |
| **Yazı** | [Her Şey Metin: PaliGemma 2 Nasıl Çalışıyor, Neyi Gösteriyor, Neyi Göstermiyor?](article/article.md) ([PDF](article/article.pdf)) |
| **Deney** | [Türkçe tabelalar ve fişlerde 224 ile 448 piksel](experiment/) |

## Kısaca

**Hangi soruyu soruyor?** Bir görsel-dil modelini iyileştirmek için daha keskin bir **göz** (yüksek çözünürlük) mü, daha büyük bir **beyin** (büyük dil modeli) mi gerekir? Makalenin katkısı, bu soruyu aynı tarifle eğitilmiş 9 modelle (3B, 10B, 28B × 224, 448, 896 piksel) kontrollü olarak sorması.

**Model nasıl çalışıyor?** SigLIP görüntüyü 14×14 piksellik parçalara bölüp görsel tokenlara çeviriyor. Tek bir doğrusal katman bu tokenları Gemma 2'nin diline taşıyor. Gemma 2 de cevabı token token yazıyor. Kutular, maskeler, tablolar ve moleküller dahil her çıktı metin.

**Ne ikna etti, ne etmedi?** Göz/beyin ayrımı sağlam: Metin ve doküman okuyan görevler çözünürlükten, çok dilli ve akıl yürütme gerektiren görevler büyük modelden kazanıyor. Buna karşılık yeni görevlerde çözünürlük ve boyut iddialarının çoğu rakamla gösterilmiyor. 28B neredeyse hiç raporlanmıyor ve göz/beyin karşılaştırması eşit bütçeli değil.

## Deneyden üç bulgu

1. **Göz, ince ayrıntıda fark yaratıyor.** Aynı 3B modelde 224'ten 448'e geçince küçük yazıda doğruluk %13'ten %39'a, fişlerde toplam tutarda %46'dan %84'e çıkıyor. Bedeli soru başına yaklaşık 3,4 kat süre.
2. **Küçük göz okuyamadığı yerde uyduruyor.** 224, Galata kitabesindeki "29 Mayıs 1453"ü "20 Mayıs 1938" diye okuyor.
3. **Türkçe soru daha iyi sonuç veriyor, ama Türkçe harfler kayboluyor.** Model `ocr` modunda Türkçe harflerin %94'ünü doğru yazıyor, soru-cevapta ise çoğunu düşürüyor ("acil cikis").

Ayrıntılar: [experiment/](experiment/)

## İçerik

| Dosya | Ne |
| --- | --- |
| [`paper.pdf`](paper.pdf) | Makale |
| [`notes.md`](notes.md) | Bölüm bölüm okuma notları |
| [`article/article.md`](article/article.md) | Yayımlanan yazı (~1.900 kelime) |
| [`article/article-long.md`](article/article-long.md) | Yazının uzun hali (~4.400 kelime, yedi yeni görevin hepsi ayrıntılı) |
| [`figures/paper/`](figures/paper/) | Makaleden alınan figür ve tablolar (CC BY 4.0) |
| [`figures/original/`](figures/original/) | Bu çalışma için üretilen görseller |
| [`experiment/`](experiment/) | Notebook, veri seti ve sonuçlar |
