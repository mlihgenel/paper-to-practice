# YZ50 Research

Yapay zekâ makaleleri üzerine Türkçe incelemeler. Her inceleme üç katmandan oluşur: makalenin bölüm bölüm okunması, makaleyi okumamış bir okur için yazılmış eleştirel bir analiz ve mümkün olduğunda makalenin iddialarını yeni bir veri üzerinde sınayan tekrarlanabilir bir deney.

## İncelemeler

| # | Makale | Analiz | Deney | Öne çıkan bulgu |
| --- | --- | --- | --- | --- |
| 01 | [PaliGemma 2](papers/01-paligemma2/)<br>Google DeepMind, 2024 | [Her Şey Metin](papers/01-paligemma2/article/article.md) | [224 ile 448 piksel, Türkçe tabelalar](papers/01-paligemma2/experiment/) | Yüksek çözünürlük küçük yazıda doğruluğu %13'ten %39'a çıkarıyor. Düşük çözünürlükte model okuyamadığı metni uyduruyor. |

## Yaklaşım

Her inceleme aynı üç soruyu cevaplar:

1. **Soru ve katkı.** Makale hangi problemi ele alıyor, alana ne ekliyor?
2. **Yöntem.** Model nasıl çalışıyor, girdiden çıktıya giden yol ne?
3. **Değerlendirme.** Hangi bulgular kanıtla destekleniyor, hangileri değil? Hangi deneyler eksik?

Deneyler, ücretsiz bir Colab GPU'sunda çalışacak şekilde tasarlanır. Veri setleri, notebook'lar ve ham sonuçlar repoda yer alır, böylece her sonuç baştan üretilebilir.

## Yapı

```text
papers/
└── NN-makale-adi/
    ├── README.md        # künye, özet ve ana bulgular
    ├── paper.pdf        # makale
    ├── notes.md         # bölüm bölüm okuma notları
    ├── article/         # analiz yazısı (yayımlanan ve uzun hali, PDF'leri)
    ├── figures/
    │   ├── paper/       # makaleden alınan figür ve tablolar
    │   └── original/    # bu inceleme için üretilen görseller
    └── experiment/      # notebook, veri seti ve sonuçlar
```

## Lisanslar ve atıf

- Makaleler ve makalelerden alınan figürler kendi lisanslarıyla kullanılır. Her görselin altında kaynağı ve lisansı belirtilir.
- Deneylerde kullanılan üçüncü taraf veri ve görseller kendi lisanslarını korur. Ayrıntılar her deneyin `data/SOURCES.md` dosyasındadır.
