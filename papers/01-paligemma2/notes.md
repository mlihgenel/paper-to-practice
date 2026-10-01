# PaliGemma 2: A Family of Versatile VLMs for Transfer

## Özet / Giriş
- PaliGemma'nın yeni versiyonu. Dil modeli Gemma 1 → Gemma 2.
- 3 boyut: 3B, 10B, 28B. 3 çözünürlük: 224, 448, 896px.
- "for Transfer": Model direkt kullanılmak için değil, bir göreve fine-tune edilmek için yapılmış.
- İki ana iddia:
  - Bazı görevler çözünürlükten, bazıları büyük LM'den daha çok kazanıyor.
  - Büyük model → daha küçük learning rate gerekiyor.
- Yeni alanlarda denenmiş: OCR, tablo, molekül, nota, uzun caption, radyografi.

## 2. Related Work
- VLM tarihi: CLIP gibi eşleştirici modeller → sıfırdan eğitilen caption modelleri → hazır görüntü kodlayıcı + hazır LM (bugünkü yöntem).
- Transfer yöntemleri: zero-shot, few-shot, fine-tuning. Makale fine-tuning'e odaklanıyor.
- Boşluk: Çözünürlük ve LM boyutunun etkisini birlikte inceleyen çalışma yok.
- Önceki çalışmalar farklı laboratuvarların farklı modellerini karşılaştırmış, yani elmayla armut. Burada ise hepsi Gemma 2, tarif aynı.
- ? "Bildiğimiz kadarıyla ilk" iddiası
- ? Görüntü kodlayıcı hiç büyütülmüyor

## 3. Model
- Mimari PaliGemma 1 ile birebir aynı. Yenilik mimaride değil.
- Parçalar: SigLIP-So400m (görüntü) → linear projection (köprü) → Gemma 2 (dil)
- Akış:
  1. Resim 14x14'lük parçalara bölünür → görsel token (224px = 256, 448px = 1024, 896px = 4096 token)
  2. Linear katman bu tokenları Gemma'nın boyutuna çevirir
  3. Görsel tokenlar + metin komutu yan yana Gemma'ya verilir (prefill)
  4. Gemma cevabı kelime kelime üretir
- Kutu ve maske de metin olarak üretiliyor: `<loc0255>` = koordinat, `<seg024>` = maske (Figure 2)
- Eğitim:
  - Aşama 0: SigLIP ve Gemma ayrı ayrı hazır alınıyor
  - Aşama 1: Birlikte 1 milyar örnek, 224px, hiçbir şey dondurulmuyor
  - Aşama 2: 448px (50M örnek) → 896px (10M örnek), OCR gibi görevlerin ağırlığı artırılıyor
  - Aşama 3: Hedef göreve fine-tuning
- Veri: Etiketler makine üretimi ama açık modellerle yapılmış, GPT-4V gibi ticari model kullanılmamış.
- Learning rate: 3B'de ×0.5, 10B/28B'de ×0.25
- Table 1 (maliyet): 3B@896 = 23.5, 28B@224 = 18.9. Yani yüksek çözünürlük, büyük modelden bile pahalı olabiliyor. Sebep: LM'nin içinden geçen görsel tokenlar.
- ? Soft-capping son aşamada kapatılmış, "bazı görevlerde kötüleştirdi" demişler ama hangileri olduğunu göstermemişler.
- ? 28B@896 farklı TPU'da eğitilmiş, maliyeti tahmini.

## 4.1 Model boyutu ve çözünürlük
- Deney: 3 model × 2 çözünürlük (224, 448), 30+ benchmark, 5 çalıştırmanın medyanı.
- Sadece learning rate taranmış, o da sadece 224px'te. 448'de aynısı kullanılmış.
- Figure 3: 3B@224'ten başlayıp ya LM büyütülmüş (3.7x maliyet) ya çözünürlük artırılmış (4.6x maliyet)
  - Çoğu görev ikisinden de kazanıyor
  - Metin/doküman/grafik/ekran görevleri → çözünürlük (resimleri zaten büyük)
  - Çok dilli + akıl yürütme görevleri (XM3600, AI2D, CountBench, NLVR2) → büyük LM
  - Kısaca: göz mü lazım, beyin mi? Göreve göre değişiyor.
- Figure 4: 3B → 10B iyi artış, 10B → 28B az ya da hiç artış yok
  - Açıklamaları: Gemma 2 2B ve 9B distill edilmiş, 27B sıfırdan eğitilmiş. Test edilmemiş, sadece hipotez.
  - Pratikte 10B tatlı nokta gibi.
- Figure 5: Büyük model → küçük learning rate
- Gemma 2 vs Gemma 1 (aynı 3B): Ortalama sadece +0.65 / +0.85 puan. Küçük fark.
- İyi: 5 çalıştırma + std var, ScienceQA doygunluğunu kendileri belirtmişler.
- ? Karşılaştırma adil değil: Çözünürlüğe %25 daha çok bütçe verilmiş. Eşit maliyetli karşılaştırma yok.
- ? LR sadece 224'te seçilmiş, bu 448'i dezavantajlı bırakmış olabilir.
- ? 896px bu analizde yok.
- ? "Çözünürlük sever" grubunun eşiği belirsiz, göz kararı gibi.

## 4.2 Text detection and recognition
- Görev: Resimdeki her kelimeyi bulmak (kutu) + okumak (metin). Çıktı: {metin, kutu} çiftleri.
- Doğru sayılma şartı: Kutu en az %50 örtüşmeli (IoU ≥ 0.5) ve metin birebir aynı olmalı. Büyük/küçük harf ve noktalama da dahil, çok katı.
- Eğitim: 7 OCR veri setinin karışımı. Test: ICDAR'15 ve Total-Text (en yaygın iki benchmark).
- Table 2: 3B@896, SOTA olan HTS'i geçiyor.
  - ICDAR'15 F1: 75.9 vs 74.5
  - Total-Text F1: 74.2 vs 72.4
- Asıl mesaj: OCR'a özel mimari yok, genel bir VLM'i fine-tune edip SOTA'yı geçiyorlar.
- Çözünürlüğü düşürünce ciddi kötüleşme var, modeli büyütünce ise iyileşme yok. 4.1'deki "metin görevleri = göz lazım" bulgusuyla tutarlı.
- Ek B.1: 15k adım, LR 1e-5, label smoothing 0.1.
- ? Farklar küçük (~1.5-2 F1) ve tek çalıştırma mı, std yok.
- ? Total-Text precision'da HTS daha iyi (75.7 vs 73.8). Kazanç recall'dan geliyor.
- ? "Çözünürlük düşünce kötüleşti, model büyüyünce değişmedi" dediler ama rakamları göstermediler.
- ? HTS aynı eğitim verisiyle mi eğitilmiş? Karşılaştırma adil mi, belirsiz.
- ? Hız yok: VLM kutuları token token üretiyor. Özel OCR modellerine göre çok yavaş olabilir.

## 4.3 Table structure recognition
- Görev: Tablo resimlerinden; tablo yapısını(satır, sütun, birleştirilmiş hücreler), hücrenin metnini, hücrenin kutusunu(bounding box) çıkarmak
- Resim -> HTML tablo yapısı 
- Etiketi bariz şekilde bozuk olan örnekler eğitim verisetinden çıkarıldı. Geçersiz tablo yapıları, üst üste binen hücre kutuları vs dahil. 
- Resimler boyutlandırılıyor
- Padding işlemi
  - Oran korunuyor. 
  - Boş kalan kısımlar beyaz piksellerle doldurulup resim kareye tamamlanıyor. 
- Ölçme/değerlendirme metrikleri: 
  - Modelin kalite değeri TEDS ve GriTS'e göre ölçülüyor. 
  - Hücre metin içeriği, hücre yapısı bounding box metrikleri ölçülüyor. 
    - S-TEDS: Sadece yapı (satır/sütun iskeleti doğru mu?)
    - TEDS: Yapı + metin birlikte 
    - GriTS-Top: Hücre topolojisi (hangi hücre nerede, kaç hücre birleşmiş)
    - GriTS-Con: Hücre içerikleri (metin)
- FinTabNet metriklerinde: 
  - PaliGemma 2 3B 896px, SOTA'yı geçiyor. 
- PubTabNet metriklerinde: 
  - PaliGemma 2 3B 896px, SOTA'yı geçiyor. (hepinde olmasa da çoğunda)
- Hücre kutuları yine <loc> tokenlarıyla kodlanıyor ve HTML'in içine gömülüyor.
  Yani model düz HTML yazıyor, koordinatlar da bunun içinde. "Her şey metin üretimi" fikri bir kez daha. Tablolar uzun olduğu için çıktı uzunluğu 4096 tokena çıkarılmış.

## 4.4 Molecular structure recognition
- Görev:  Bir molekül çizimine bakıp onu SMILES formatına çevirmek.
- Görev bize graf yapısındaki molekülleri metne çevirmek
- Eğitim veriseti: PubChem'den 1 milyon molekül resmi 
- Test: MolScribe'ın kullandığı 5.7 bin resim. Bunlar da sentetik ama farklı bir araçla (ChemDraw) çizilmiş. 
- Tam eşleşmede MolScribe'dan daha iyi sonuç elde edilmiş(PaliGemma 2 10B @ 448px) 
  - MolScribe -> 93.8 | PaliGemma -> 94.8 (Exact Match)
- Ama OCR için -> 3B@896px, Tablolar için -> 3B@896px, Molekül için -> 10B@448px 
  - Çözünürlüğü artırmak işe yaramamış 
    - Artırmak yeterli değil denmiş ama bunun rakamları, ölçümleri verilmemiş. 
- Peki neden 10B seçildi? Neden 3B ya da 28B seçilmedi, bunların karşılaştırmaları, metrikleri paylaşılmamış 
  - 10B, 3B'den ne kadar iyi belli değil
- Test verisi tamamen sentetik verilerden oluşuyor. 

## 4.5 Optical music score recognition
- Görev: Nota kağıdının resmini dijital nota dosyasına çevirmek. OMR (Optical Music Recognition)
- Çıktı formatı olarak kern kullanılıyor. 
  - **kern formatı: müziği metin olarak yazmak; her notanın perdesi(do, re, mi…), süresi (dörtlük, sekizlik…), ayrıca artikülasyon işaretleri ve ölçü çizgileri
- Resim -> metin (**kern formatı). Moleküllerde SMILES, tablolarda HTML, müzik notalarında **kern 
- Veriseti olarak: GrandStaff (53.7 bin resim)
- Değerlendirme metrikleri: 
  - CER (Character Error Rate): Karaktere seviyesinde hata 
  - SER (Symbol Error Rate): Sembol seviyesinde hata. Bir nota birkaç karakterle yazılıyor, sembol bunların birleşimi 
  - LER (Line Error Rate): kern satırı seviyesinde hata. En katı olanı.
  - Büyükten küçüğe: karakter -> sembol -> satır 
- PaliGemma 2 3B @ 896px, Bu metriklerde Sheet Music Transformer (önceki en iyi) modeline göre çok önde. 
  - CER -> (SheetMusic: 3.9 | PaliGemma 2: 1.6)
  - SER -> (SheetMusic: 5.1 | PaliGemma 2: 2.3)
  - LER -> (SheetMusic: 13.1 | PaliGemma 2: 6.7) 
  - değerlerin küçük olması istediğimiz bir şey (hata oranı olduğu için)
- Notalar sadece tek satır olacak şekilde kullanılmış. Gerçek nota sayfasında birden çok satır var. Tam sayfa denemesi yapılmamış
- Yine çözünürlük ve model boyutu rakamları yok. En iyi hata oranının 896px'de olduğu yazılmış ama diğer çözünürlüklerdeki hata oranları belirtilmemiş
- Yine aynı şekilde model boyutunu 3B'den 10B'ye çıkarmanın error oranı düşürmede etkisi olmadığı söylenmiş ama rakamlarla desteklenmemiş ve 28B modeli hiç denenip sonuçları açıklanmamış 
   

## 4.6 Long captions
- Görev: Bir resme bakıp uzun ve ayrıntılı bir açıklama yazmak. Kısa bir "bir at duruyor" cümlesi değil, 7-8 cümlelik bir paragraf.
- Neden önemli: Text-to-image modellerini (Stable Diffusion gibi) eğitmek için bu tür detaylı açıklamalar lazım. Açıklama ne kadar ayrıntılı olursa üretilen resim de o kadar kontrol edilebilir oluyor.
- Eğitim veriseti: DOCCI  
  - 15bin resim var ve açıklamaları insan tarafından yazılmış(sentetik değil)
  - bir açıklama ortalama 7.1 cümle. yaklaiık 639 karakter ve 136 kelime.
  - Açıklamalarda nesnelerin birbirine göre konumu, sayma, resimdeki yazılar ve dünya bilgisi gibi şeyler var.
- NES metriği kullanılarak ölçüm yapılıyor. 
  - Otomatik değerlendirme değil, insanlar tarafından yapılıyor. 
  - Üretilen her cümle ayrı ayrı gösterilip "bu cümle resimle uyuşuyor mu?" diye soruluyor. 4 seçenek var: Entailment (doğru), Neutral (resimden anlaşılmıyor), Contradiction (yanlış), Nothing to assess (değerlendirilecek bir şey yok).
  - NES (Non-Entailment Sentences): Neutral ve Contradiction cümlelerin oranı. Kısaca uydurma veya doğrulanamayan cümle yüzdesi. Düşük olması iyi.
  - Her cümleye 5 kişi bakıyor ve çoğunluğun kararı geçerli sayılıyor 
- Sonuçlar
  - PaliGemma 2 10B: NES 20.3, tablodaki en iyi değer.
  - PaliGemma 2 3B: 28.4
  - PaliGemma 1 3B: 34.3. Aynı boyutta Gemma 1'den Gemma 2'ye geçince yaklaşık 6 puan iyileşme var.
  - Diğer modeller (LLaVA 40.6, InstructBLIP 42.6, MiniGPT-4 52.3) çok daha fazla uyduruyor.
  - Cümle uzunlukları birbirine yakın (PaliGemma 2 yaklaşık 520 karakter, PaliGemma 1 535). Yani PaliGemma 2 daha kısa yazdığı için daha az hata yapıyor değil.
  - Figure 10'da (Ek) aynı at resmi için bütün modellerin yazdığı açıklamalar var, uydurma cümleler kırmızıyla işaretlenmiş. Bakmaya değer; özellikle VILA'nın "kuyruğu hafifçe sallanıyor" gibi süslü uydurmaları dikkat çekiyor.
- Öncdeki bölümlerde en iyi sonuçları 3B@896 veriyordu (molekül hariç), burada. 3B->10B yapınca belirgin bir artış oldu. NES: 28.4 -> 20.3 
- DOCCI'de fine-tune edilen modeller yalnızca Pali ailesi (PaLI-5B, PaliGemma 1 ve 2). LLaVA, VILA gibi modeller DOCCI'de hiç eğitilmemiş, genel instruction-tuned haliyle kullanılmış. Adil olan tek karşılaştırma PaliGemma 1 ile PaliGemma 2 arasındaki. 
- "Çözünürlük de iyileştiriyor" demişler ama tabloda sadece 448px var. 224 ve 896 rakamları gösterilmemiş. 4.3-4.5'te de gördüğümüz "söyleyip göstermeme" durumu burada da var. 
- 3B ve 10B boyutlarınin sonucu var ama yine 28B için herhangi bir metrik yok! Model büyüdükçe NES değerinin dştüğünü gözlemliyoruz ama 28B olduğunda da bu düşüş devam ediyor mu göremiyoruz. 

## 4.7 Spatial reasoning
- PaliGemma nesne bulma, segmentasyon ve "şu nesneyi göster" gibi görevlerde zaten iyi 
- Ama benchmarklarda sorun var: 
  - etiketler çoğunlukla makineler tarafından üretilmiş 
  - zor hataları. yakalayamıyorlar (örn: kedi masanın üzerinde değil)
- VSR (Visual Spatial Reasoning) benchmarkını kullanıyorlar. 
- Modele bir resim ve cümle veriliyor 
  - model bu cümlenin doğru mu yanlış mı olduğuna karar veriyor(sınıflandırma)
  - PaliGemma'da her şey metin olduğu için true ya da false yazıyor. 
- iki adet test spliti var: 
  - Random split: train ve test rastgele bölünmüş, daha kolay 
  - Zero-shot split: tessteki nesneler/kavramlar eğitimde hiç görülmemiş, daha zor ve genelleme yeteneğini ölçüyor 
- Model boyutu 3B -> 10B yapınca her iki splitte de artış var (3B@224: 74.8 -> 81.6, 10B@224: 79.8 -> 86.8)
- Karşılaştırılan modeller eski. Fine-tune edilmiş tek rakip LXMERT ve 2019'dan. Güncel ve fine-tune edilmiş bir VLM ile karşılaştırma yok. "Önceki fine-tuned modelleri geçtik" iddiası bu yüzden zayıf kalıyor.
- InstructBLIP zero-shot, PaliGemma fine-tuned. Bunları karşılaştırmak adil değil, yazarlar da bunu zaten "fine-tuning kazandırıyor" diye sunuyor. 
- 224px üstü işe yaramadı demişler ama rakamlarla desteklenmemiş yine.
- 3B ve 10B karşılaştırması yapılıp 28B yine karşılaştırmaya dahil edilmemiş ve verileri paylaşılmamış 
- Ek B.6'daki hiperparametre taraması çok geniş. LR, weight decay, dropout ve epoch değerlerinin kombinasyonu yaklaşık 216 ayrı deneme ediyor. En iyisinin hangi veri üzerinde seçildiği belirtilmemiş. Test üzerinde seçildiyse sonuç şişmiş olabilir. Std de yok. 

## 4.8 Radiography report
- Görev: Göüğs röntgenine bakıp radyoloğun yazacağı raporu üretmek 
- bu makalede bu kısım ayrı bir tıbbi alan olarak görülmüyor, "röntgen resmi üzerinde uzun caption" olarak ele alınıyor. 
- Veriseti: MIMIC-CXR 
  - 377bin röntgen resmi var. 
  - her resimle birlikte radyoloğun serbest metin olarak yazdığı rapor var 
  - Train/val/test bölümü Flamingo-CXR ile aynı. Bu iyi, çünkü karşılaştırma aynı veri üzerinde yapılıyor. 
- Rapor yapısı: 
  - INDICATION: Röntgen neden çekildi, örneğin: nefes darlığı mı var?, ödem mi? 
  - FINDINGS: resimde neler görülüyor  
  - IMPRESSION: radyoloğun yorunu ve özeti 
  - Modele indication kısmı hazır olarak veriliyor, findings ve impression kısımlarını model yazıyor. Bu, gerçek klinikteki işleyişi taklit ediyor: radyolog da neden çekildiğini bilerek raporu yazar. 
- Raporlarda sık sık "önceki röntgene göre kötüleşmiş" gibi ifadeler geçiyor. Model önceki röntgeni görmediği için bu ifadeleri öğrenirse uydurmak zorunda kalır.
- Bu yüzden bu kısımlar Gemini 1.5 Pro ile raporlardan silinmiş.
- Ölçüm: RadGraph F1
  - Hem üretilen rapordan hem de gerçek rapordan tıbbi varlıklar çıkarılıyor. Örneğin "plevral efüzyon: var", "ödem: yok".
  - Sonra bu varlıklar birbiriyle karşılaştırılıyor. Bir bulgunun var mı yok mu dendiği ve bulgular arasındaki ilişkiler de hesaba katılıyor.
  - Bu klinik bir metrik. Buna ek olarak CIDEr, BLEU ve ROUGE gibi klasik metin benzerliği metrikleri de raporlanmış.
  - RadGraph F1'de SOTA elde edilmiş: 24.4'ten 29.5'e.
  - Burada da ilginç olan şu: Med-Gemini tıbba özel bir model, PaliGemma 2 ise genel bir VLM. Yine de fine-tune edilince tıbba özel modeli geçiyor. Bu, OCR bölümündeki mesajın aynısı.
  - Çözünürlük ve model boyutunun ikisi de "biraz" iyileştirmiş.
- Test verisinin eğitim ve hiperparametre ayarı sırasında hiç kullanılmadığını açıkça yazmışlar. 4.6'da seçim aynı veri üzerinde yapılmıştı, burada ise yapılmamış. 
- Hiçbir radyolog raporları değerlendirmemiş. Değerlendirmenin tamamı otomatik metriklerle yapılmış. Tıpta asıl önemli olan soru "bu rapor klinikte kullanılabilir mi?" 
- Çözünürlük ve boyut etkisi yine eksik gösterilmiş. "Biraz iyileştirdi" demişler ama tabloda sadece 896px var. 224 ve 448 sonuçları yok, 28B de yok. 
- Veri tek bir hastaneden geliyor. Farklı bir hastanede, farklı bir röntgen cihazında veya farklı bir hasta profilinde modelin nasıl çalışacağı bilinmiyor. 
- Gemini ile yapılan temizlik test raporlarına da uygulandı mı? Belirtilmemiş. Test referansları değiştirildiyse, baseline'ların sonuçları temizlenmemiş raporlar üzerinde hesaplanmış olabilir. Bu durumda karşılaştırma adil olmaz. Ayrıca bir LLM'in tıbbi metni düzeltirken hata yapıp yapmadığı da kontrol edilmemiş. 

## 4.9 CPU inference and quantization
- Soru: GPU/TPU olmadan, sadece CPU'da çalışır mı? Ne kadar hızlı, kalite düşer mi?
- Araç: gemma.cpp. Hafif bir C++ çıkarım motoru, 8-bit quantization destekliyor. (Alternatifler: llama.cpp, XNNPack)
- Kurulum: 3B@224, COCOcap'te fine-tune edilmiş. Tek resim, "describe this image" komutu.
  - Girdi: 256 görsel + 4 metin = 260 token (prefill)
  - Çıktı: 11 token ("A large building with two towers on the water")
  - Batch size 1
- Hız (Table 9):
  - ViT: Resmi SigLIP'ten geçirme
  - Prefill: 260 tokenı Gemma'ya okutma. En yavaş kısım.
  - Extend: Cevabı token token üretme
  - Apple M1 Max (laptop): ViT 1.6s + prefill 8.2s + extend 0.9s, toplam ~10 saniye
  - AMD Genoa 32 çekirdek (sunucu): toplam ~1.2 saniye
  - Yani laptopta çalışıyor ama yavaş, güçlü sunucu CPU'sunda makul.
- Kalite (Table 10): Jax F32 (12.1GB) vs gemma.cpp quantized (4.0GB)
  - 5 görev: COCOcap, TextCaps, AI2D, OKVQA, DocVQA
  - Göreceli skorlar %99.9 - %100.2 arası. Yani fark yok.
  - Bellek 3 kat küçülüyor, kalite aynı kalıyor.
- f32 → bf16 geçişinin kayıpsız olduğu zaten PaliGemma 1'de gösterilmişti.
- Pratik mesaj: Model laptopta/sunucu CPU'sunda, küçültülmüş haliyle kullanılabilir.
- Sadece en küçük ayar (3B@224) denenmiş. 896px'de 4096 görsel token var, prefill ~16 kat uzar. 10B/28B hiç yok.
- Tek resim, 11 tokenlık kısa cevap. Uzun caption (4.6) veya rapor (4.8) gibi uzun çıktılarda süre çok daha fazla olur.
- Hız ölçümünde tekrar/ortalama yok gibi, tek çalıştırma.
- GPU ile hız karşılaştırması yok, "CPU ne kadar yavaş" sorusunun referansı eksik.
- Quantization kalitesi de sadece 224px ve kısa çıktılı görevlerde test edilmiş. Uzun çıktıda hata birikiyor mu, belli değil.
- "8-bit switched-floating-point" detayı açıklanmamış, hangi katmanların quantize edildiği belirsiz ("mixed").
- 

## 5. Sonuç
- İddialar:
  - Açık ağırlıklı, farklı boyut ve çözünürlüklerde bir model ailesi.
  - Caption, VQA ve video görevlerinde güçlü transfer.
  - Büyük varyantlar (10B, 28B), bütçesi olanlar için PaliGemma 1'e göre ciddi iyileşme.
  - PaliGemma 1'de olmayan alanlarda da iyi: müzik, molekül, tıbbi görüntüleme.
- Girişteki iki ana iddia (göz mü beyin mi, büyük model → küçük LR) sonuçta tekrar edilmemiş. Analiz makalesinden çok model tanıtım raporu gibi bitiyor.
- Görevlere göre en iyi ayar (makale toparlamıyor, bölümler yan yana konunca çıkıyor):
  - Göz grubu (3B@896): OCR, tablo, nota
  - Beyin grubu (10B): molekül, uzun caption, spatial reasoning
  - İkisi de az katkı: radyografi
- "Büyük varyantlar ciddi iyileşme" iddiası kendi bulgularıyla tam örtüşmüyor: 10B → 28B kazanç az/yok (4.1), özel görevlerde 28B hiç yok.
- "Tıbbi görüntülemede başarılı" fazla iddialı: radyolog değerlendirmesi yok, RadGraph F1 ~29.5.
- Sınırlılıklar ve gelecek çalışma kısmı yok. Eksik deneylerden hiç bahsedilmiyor.
- Video görevleri ana metinde hiç işlenmemiş, sadece ekteki tablolarda var.

## Medium fikirleri
-
