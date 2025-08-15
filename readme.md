# TRADE-AI

Kurulum

* Python 3.10+
* pip install -r requirements.txt
* .env oluştur:
  TELEGRAM\_API\_ID=...
  TELEGRAM\_API\_HASH=...
  EXCHANGE\_ID=binance
  TIMEFRAME=1m
  TRADING\_SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT

Çalıştırma

* python app.py





şimdi yapımız şu şekile diimi? analysts,news,policy,whales dosyalarından aldığımız verilerle birlikte ohlc verileri ve indicatorlerin tümünü kullanarak ai mizi eğitiyoruz o da bize momentum oluştuğu anda oluşan momentumun % kaç ilerleme ihtimalini veriyor böylece bizde geriye kalan % momentuma göre strateji belirleyip ona göre işlem yapıyoruz bu arada ai miz de daimi olarak yaptığımız işlemledeki momentumun % kaç ilerleme ihtimalini bilme oranını ve karlılık sonucuna    göre gerektiğinde yeni indikatörler keşfedip kendi yazılımını geliştirip kendini evolve edicek
Assistant Icon
Evet, doğru özetlediniz. Sistem mimarisi şu şekilde çalışıyor:

Veri katmanı
Piyasa: OHLCV + orderbook (spread%, volume\_ratio, vb.).
Dış dünya: whales (Telegram), news, policy, analyst (şu an iskelet; genişletilebilir).
Feature set: Teknik indikatörler + mikro yapı + whale\_\* + news\_\* + policy\_\* + analyst\_\* + ATR.
Algılama ve karar akışı (canlı)
Momentum tespiti: Basit kuralla tetik (ör. son 5 bar ≥ 1.5% ve volume\_ratio > 1.5).
Rejim tahmini: RegimeClassifier → trend\_up/down/range/volatile olasılıkları.
Continuation tahmini: ContinuationModel → cont\_prob (devam olasılığı) + remain\_atr + expected\_move\_pct.
Fake momentum gating: FakeMomentumDetector → fake\_p ile pozisyonu küçült/iptal etme.
Strateji seçimi: choose\_strategy(reg\_probs, cont, side) → trend/meanrev/breakout, boyut çarpanı, SL/TP ATR, tutma süresi.
Uygulama: Sinyal varsa boyut ve risk parametreleri ayarlanır (şu an “işlem gönderme” kısmı örnek niyetinde).
Öğrenme döngüsü (online)
MomentumEventTracker: Olay anında feature vektörünü kaydeder.
Lookahead sonunda (örn. 30 dk): “fake/real” etiketlenir ve FakeMomentumDetector.partial\_fit ile güncellenir.
Not: Continuation/Regime modelleri offline eğitilir (scripts/train\_\*); canlıda sadece inference yapılır.
Eğitim/yeniden eğitim (offline)
Veri toplama: app çalışma esnasında df + dış dünya feature’larını CSV/Mongo gibi bir depoya per-bar ekleyin (öneri).
Rejim eğitimi: scripts/train\_regime\_classifier.py → models/regime\_clf.joblib
Continuation eğitimi: scripts/train\_continuation.py → models/cont\_cls.joblib, models/cont\_reg.joblib
Yeni modelleri “shadow” modda doğrulayın; iyi ise prod’a alın.
Evolve katmanı (ilerleyen aşama)
Yeni indikatör keşfi: core/indicator\_discovery.py (skeleton) → auto feature search; en faydalı olanı otomatik olarak feature set’e ekleme.
Kendi kodunu yazan sistem: core/self\_modifying\_engine.py (skeleton) → strateji şablonları üreten/geliştiren motor; A/B ve güvenlik sandbox’ı ile.
AutoML/NAS: core/automl\_nas.py (skeleton) → rejim/continuation modellerinin mimari/hyperparam aramasını otomatikleştirme.
Adversarial ve swarm modülleri: core/adversarial\_training.py, core/swarm\_trading.py (skeleton) → sağlamlık ve keşif.
Neden bu düzen güvenli?

Whale/news/policy/analyst verileri ağırlıkla eğitimde kullanılıyor; canlıda sadece tahmini güçlendiriyor, geç kalmış “onay”a aşırı ağırlık vermiyoruz.
Canlı kararın ana sürücüsü: continuation olasılığı ve rejim olasılıkları.
Fake momentum filtresi yüksek riskli anları kapatıyor/küçültüyor.
Dikkat edilmesi gerekenler

Veri sızıntısı yok: Eğitimde geleceği kullanan etiketler (MFE/Sharpe/rejim label) sadece offline.
Feature “freshness”: Telegram/News gibi dış sinyaller gecikmeli olabilir; time alignment yapın (timestamp’ları UTC ve bar kapanışıyla hizalayın).
Değerlendirme: Time-based split ve son dönem performansına ağırlık verin; kalibrasyon (CalibratedClassifierCV) kullanın.
Promosyon kriteri: Yeni model sadece canlı shadow’da belirli metrikleri (cont Brier score, rejim logloss, PnL/Sharpe uplift) geçerse canlıya alın.
Kısa görev listesi (sonraki adımlar)

Logger ekleyin: app.loop içinde her bar için bir satır CSV (timestamp, symbol, close, tüm feature’lar, sinyal/sonuç meta).
İlk dataset ile rejim/continuation modellerini eğitip models/ içine koyun.
News/Policy/Analyst modüllerini doldurun (CryptoPanic/NewsAPI/GDELT; düzenleyici takvikleri için takvim/keyword tarama).
IndicatorDiscoveryEngine’i etkinleştirin: en iyi 3 yeni feature’ı seçip train scriptlerine dahil edin; A/B kıyaslayın.
Strateji yürütme katmanı: SL/TP ATR tabanlı risk yönetimini gerçek emirle bağlayın (paper trading API).
Sonuç

Evet: Çok-kaynaklı verilerle eğitilen AI canlıda momentum anında “devam olasılığı ve kalan potansiyel”i verir; rejim olasılıklarıyla birlikte strateji seçimi ve boyutlandırma yapılır.
AI çevrim içi olarak fake momentum tespitinde kendini günceller; periyodik offline yeniden eğitimle de rejim/continuation iyileşir.
İleride otomatik indikatör keşfi ve kendi kodunu yazan strateji üretimi modülleriyle sistem evrimsel hale gelir.

