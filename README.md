**JAIDE** (**KGRU**) – Teljes dokumentáció (Magyar)

> Forrás: [https://deepwiki.com/kollarsandor/kissofdeath](https://deepwiki.com/kollarsandor/kissofdeath) > Generálva: **2026**-04-03

---

 Projekt Áttekintés

A **JAIDE** (v40) egy gyökérszintű **LLM** rendszer, amely a **KGRU** (Kalmár László - Gábor Dénes - Riesz Frigyes unity) architekturális filozófiára épül. Egy vertikálisan integrált stacket képvisel, amely magában foglal egy egyedi neurális architektúrát, egy dedikált relációs motort, hardveres gyorsító kerneleket és egy átfogó formális verifikációs csomagot.

A projektet elsősorban Zig nyelven implementálták, a teljesítménykritikus kerneleket Futhark nyelven, a formális specifikációkat pedig Lean 4, Cryptol és **SAW** segítségével.

Rendszercélok és Filozófia A **JAIDE** célja, hogy túllépjen a standard transformer architektúrákon a következők használatával: Relational Signal Flow (**RSF**): Egy neurális réteg stack, amely a relációs adatmozgást hangsúlyozza. Morpheme-Guided Tokenization (**MGT**): Egy tokenizer, amely nyelvi struktúrát és morféma-felbontást használ az egyszerű byte-párok helyett. Stochastic Fractal Descent (**SFD**): Egy egyedi optimalizációs stratégia a paraméterfrissítésekhez, amely momentum becslésen alapul. Formális Szigor: Minden alapvető komponenst matematikai bizonyítások támasztanak alá, hogy biztosítsák a biztonságot és a helyességet a nagy tétre menő érvelési feladatokban.

A **KGRU** név magyar úttörők előtt tiszteleg: Kalmár László (logika/verifikáció), Gábor Dénes (információelmélet/jelterjedés) és Riesz Frigyes (funkcionálanalízis/kvantumlogika).

Magas Szintű Architektúra A rendszer három fő síkra oszlik: az ML Pipeline, a Core Relational Engine és az Acceleration Layer.

Rendszer Adatfolyam: Szövegtől az Érvelésig

graph TD Input[*User Natural Language*] --> **MGT**[***MGT** Tokenizer (src/tokenizer/mgt.zig)*] **MGT** --> **RSF**[***RSF** Processor (src/processor/rsf.zig)*] **RSF** --> **SSI**[***SSI** Index (src/index/ssi.zig)*] **SSI** --> Ranker[*Ranker (src/ranker/ranker.zig)*]  subgraph *Core_Relational_Engine* Ranker --> **NSIR**[***NSIR** Graph (src/core_relational/nsir_core.zig)*] **NSIR** --> RO["ReasoningOrchestrator (src/core_relational/reasoning_orchestrator.zig)*] RO --> **ZRT**[*ZRuntime (src/core_relational/z_runtime.zig)*] end

subgraph *Acceleration_and_Storage* **RSF** -.-> Futhark[*Futhark Kernels (src/hw/accel/futhark_kernels.fut)*] **SSI** -.-> **RTL**[***SSI** Search **RTL** (src/hw/rtl/SSISearch.hs)"] end

Főbb Alrendszerek 1. ML Pipeline Az ML pipeline kezeli a nyers szöveg nagy dimenziós vektorokká történő átalakítását és vissza. **RSF** Processor: Kezeli a neurális hálózati rétegeket (LayerCore) és a súlymátrixokat (s_weight, t_weight). **MGT** Tokenizer: Kezeli a szótárat és a morféma-felbontást, támogatva a magyar és angol nyelvet. **SSI** & Ranker: Tömör szemantikus indexelést és **LSH**-alapú visszakeresést biztosít a kontextuskezeléshez.

2. Core Relational Engine Ez a **JAIDE** *agya*, amely túllép az egyszerű következő-token jósláson a strukturált érvelés felé. **NSIR** Graph: Egy önhasonló relációs gráf, amely a tudást Node és Edge típusokon keresztül ábrázolja. Quantum Logic: Egy réteg a valószínűségi relációs érveléshez, amely qubit reprezentációkat és kvantumkapukat használ. Z-Runtime: A relációs logika és a ZVariable kezelés végrehajtási környezete.

3. Hardver és Disztribúció A **JAIDE**-t nagy teljesítményű végrehajtásra tervezték különböző backendeken. Futhark Kernels: Párhuzamosított tensor műveletek **GPU**-khoz. **RTL** Modules: Az **SSI** és a Ranker hardverszintű implementációi (Clash/Haskell) **FPGA**/**ASIC** telepítéshez. Distributed Training: Többcsomópontos koordináció **NCCL** használatával és Modal cloud integráció **B200** **GPU**-khoz.

 Repository Szervezés
| Könyvtár | Cél | Főbb Kód Entitások |
| :--- | :--- | :--- |
| src/core/ | Alapvető primitívek | Tensor, Memory, IO |
| src/processor/ | Neurális architektúra | RSF, LayerCore, OFTB |
| src/core_relational/ | Érvelési motor | NSIR, ZRuntime, ReasoningOrchestrator |
| src/hw/ | Hardveres gyorsítás | Futhark, RTL (Clash), CUDA |
| src/verification/ | Formális bizonyítások | Lean 4, Viper, SAW, Agda |
| src/api/ | Telepítés | InferenceServer |

---

 Első Lépések: Build és Konfiguráció

Ez az oldal a **JAIDE** (v40) **KGRU** rendszer építésének, konfigurálásának és telepítésének technikai specifikációit tartalmazza.

Toolchain és Követelmények A **JAIDE** a Zig programozási nyelv használatával épül, és az **LLVM**-et használja a formális verifikációhoz és a bitkód generálásához. Zig: A 0.13.0-s verzió a minimálisan szükséges. Futhark: Szükséges a **GPU** gyorsító kernelek generálásához. Nix/Replit: A környezetet a Nix kezeli.

Build Targetek A projekt a Zig Build System-et (build.zig) használja a több végrehajtási target kezelésére.

| Target Név | Bináris Név | Forrás Gyökér | Cél |
| :--- | :--- | :--- | :--- |
| jaide | jaide | src/main.zig | Standard CPU/GPU interaktív és betanítási mód. |
| jaide-gpu | jaide-gpu | src/main_gpu.zig | Dedikált GPU-optimalizált végrehajtási útvonal. |
| jaide-distributed | jaide-distributed | src/main_distributed.zig | Többcsomópontos betanítás koordinációja. |
| jaide-distributed-futhark | jaide-distributed-futhark | src/main_distributed_futhark.zig | Futhark-gyorsított elosztott betanítás. |
| jaide-inference-server | jaide-inference-server | src/inference_server_main.zig | HTTP API a modell kiszolgálásához. |

Konfiguráció és **CLI** Flagek A MainConfig struct definiálja a rendszer működési konstansait: Dimenziók: Az alapértelmezett embedding dimenzió **128**, a maximum **16384**. Architektúra: Alapértelmezés szerint 4 **RSF** réteg. Optimalizáció: Alapértelmezett learning rate 0.**001** és momentum 0.9.

Cloud Telepítés (Modal) A **JAIDE** a Modal-t használja a nagy teljesítményű felhős végrehajtáshoz, kifejezetten az elosztott betanításhoz **NVIDIA** **B200** **GPU**-kon.

---

 Rendszerarchitektúra Áttekintés

A rendszer egy pipeline-on keresztül dolgozza fel az adatokat, amely a nyers szöveget többdimenziós tensorokká, majd relációs hármasokká, végül pedig egy kereshető szemantikus indexszé alakítja.

## Ingesztió: A nyers szöveget az MGT (Morpheme-Guided Tokenizer) dolgozza fel.

## Feldolgozás: A tokenek embeddingekké alakulnak, és áthaladnak az RSF (Relational Signal Flow) rétegeken. ## Optimalizáció: A gradienseket az SFD (Stochastic Fractal Descent) optimizer kezeli. ## Indexelés: A feldolgozott jeleket az SSI (Succinct Semantic Index) tárolja. ## Relációs Logika: A CREVPipeline relációs hármasokat von ki az NSIR (Self-Similar Relational Graph) számára. ## Visszakeresés: A Ranker LSH-alapú pontozást használ a legrelevánsabb kontextus lekéréséhez az indexből.

---

 Alapvető Primitívek

Ez a szakasz a **JAIDE** rendszer alapjául szolgáló adatszerkezeteket és segédprogramokat mutatja be.

Tensor: Többdimenziós Tömb Motor A Tensor struct a numerikus számítások elsődleges adatszerkezete. Többdimenziós adatokat kezel egy Shape absztrakció segítségével. Reference Counting: A tensorok referenciaszámlálást használnak a memória kezelésére. Copy-on-Write (CoW): A szeletelési vagy nézetműveletek teljesítményének optimalizálása érdekében a tensorok CoW szemantikát valósítanak meg.

Memóriakezelés A **JAIDE** egy sor egyedi allokátort valósít meg a töredezettség minimalizálása és a teljesítmény maximalizálása érdekében. Arena és ArenaAllocator: Gyors, tömeges allokációkhoz. PoolAllocator: Fix méretű blokk allokációhoz. SlabAllocator: Hatékony allokáció különböző kettő hatvány méretekhez. A rendszer szigorúan betartja a secureZeroMemory házirendet, biztosítva, hogy minden érzékeny adat törlődjön a memóriából a felszabadításkor.

I/O, Típusok és Modell Szerializáció A rendszer egyedi típusokat definiál a pontosság és a teljesítmény biztosítása érdekében, beleértve a fixpontos matematikát (Fixed32_32). A modell szerializációs logikája kezeli a **JAIDE** modellek bináris formátumát, specifikus magic számokat használva a fájltípusok azonosítására.

---

 **LLM** Pipeline: Tokenizer, Processor, Optimizer, Index és Ranker

**RSF** Processor (Relational Signal Flow) Az **RSF** Processor a rendszer neurális gerince. Relational Signal Flow rétegek stackjét valósítja meg, amelyek specializált Gated Recurrent Unit-ok a relációs modellezéshez. Minden LayerCore állapotot tart fenn az s_weight (forrás) és t_weight (cél) mátrixokon keresztül.

**MGT** Tokenizer (Morpheme-Guided Tokenization) Az **MGT** Tokenizer túllép a standard **BPE**-n a morféma-felbontás beépítésével. Azonosítja a prefixumokat, szuffixumok és gyököket a komplex agglutináló nyelvek (mint a magyar) kezeléséhez.

**SFD** Optimizer (Stochastic Fractal Descent) Az **SFD** Optimizer valósítja meg a betanítási logikát. Egy egyedi *Stochastic Fractal Descent* algoritmust használ, amely magában foglalja a momentum becslést és a gradiens vágást. Natív támogatást nyújt a többprecíziós kvantáláshoz (fp4, fp8, fp16, fp32).

**SSI** Index (Succinct Semantic Index) Az **SSI** Index biztosítja a szemantikus szegmensek nagy sebességű visszakeresését. Egy trie-alapú hash faként van implementálva, amely Segment adatokat tárol.

Ranker (**LSH**-alapú Eredmény Rangsorolás) A Ranker a visszakeresési pipeline utolsó szakasza, amely pontozza az **SSI** által visszaadott jelölt szekvenciákat. Locality Sensitive Hashing (**LSH**) és N-gram átfedési súlyokat használ.

---

 Core Relational Engine

A Core Relational Engine (**CRE**) a **JAIDE** elsődleges érvelési és tudásreprezentációs alrendszere. Egy Self-Similar Relational Graph (**NSIR**)-t használ Kvantumlogikával kombinálva az információk ábrázolására és manipulálására.

**NSIR** Gráf és Kvantumlogika Minden Node a gráfban tartalmaz egy Qubit állapotot, lehetővé téve a rendszer számára a bizonytalanság és a szuperpozíció ábrázolását a tudásállapotokban. A kapcsolatokat Edge objektumok definiálják egy EdgeQuality-vel (pl. superposition, entangled, fractal).

**ESSO** Optimizer és Chaos Core Az **ESSO** (Entangled Stochastic Symmetry Optimizer) felelős a gráf szerkezeti integritásának és hatékonyságának fenntartásáért. SymmetryGroup transzformációkat használ a minták azonosítására. A Chaos Core a motor *idegrendszereként* működik, biztosítva egy DynamicTaskScheduler-t és egy ContentAddressableStorage-t.

Reasoning Orchestrator A Reasoning Orchestrator kezeli a magas szintű logikát, az érvelést három ThoughtLevel kategóriába osztva: local, global és meta.

---

 Kvantumszámítógép Integráció

A Kvantumszámítógép Integrációs réteg biztosítja a hidat a **JAIDE** relációs gráf (**NSIR**) és a fizikai kvantumprocesszorok vagy szimulátorok között. Hardware Abstraction: Definiálja a különböző **IBM** Quantum backendek (Heron, Eagle, Falcon) fizikai jellemzőit. Cloud Integration: Implementálja a **HTTP** klienst az **IBM** Quantum **API**-hoz. Task Adaptation: Elemzi a gráfot, hogy kivonja a magas összefonódású klasztereket, és végrehajtható kvantumfeladatokká alakítsa őket.

---

 Hardveres Gyorsítás

A hardveres gyorsítás a **JAIDE** rendszerben egy többszintű stack, amelyet a számításigényes tensor műveletek és keresési algoritmusok specializált hardverre történő kiszervezésére terveztek.

Futhark Gyorsító Réteg A Futhark réteg adatpárhuzamos **GPU** kerneleket biztosít az **RSF** processzorhoz és az **SFD** optimizerhez. Egy Zig-to-C **FFI**-t használ a **GPU** memória és a kernel végrehajtás kezelésére.

**RTL** Hardver Modulok (Clash/Haskell) A rendszer eredeti **RTL** dizájnokat tartalmaz Clash nyelven írva. Ezek a modulok az **SSI** Index alapvető keresési és rangsorolási logikájának gyorsítására szolgálnak.

**FPGA** és **ASIC** Implementáció Az **FPGA** réteg integrálja a generált **RTL**-t egy fizikai bitstreambe. Egy **AXI4**-Lite regisztertérképet definiál. A termelési szintű teljesítmény érdekében a kódbázis tartalmaz egy fizikai tervezési folyamatot, amely a **TSMC** 28nm-es folyamatcsomópontot célozza meg Synopsys eszközök használatával.

---

 Elosztott Betanítás

Az elosztott betanítási infrastruktúra biztosítja a szükséges komponenseket a modell betanításának skálázásához több **GPU**-n és csomóponton keresztül.

Distributed Trainer és **GPU** Coordinator A GPUCoordinator a lokális hardverkezelés központi hatósága. Felelős a **CUDA** eszközök felsorolásáért és az **NCCL** kommunikátorok inicializálásáért. A DistributedTrainerFuthark a koordinátort használja egy adatpárhuzamos betanítási stratégia megvalósításához.

**NCCL** Kötések és Modal Cloud Telepítés A **JAIDE** az **NVIDIA** hardverrel egy sor **FFI** kötésen keresztül lép kapcsolatba az **NCCL**-hez és a **CUDA**-hoz. A felhős skálázást a ModalGPUClient kezeli, amely a Modal **API**-val kommunikál a betanítási feladatok telepítéséhez nagy teljesítményű hardvereken (pl. **B200**).

---

 Inference Server **API**

Az Inference Server egy **HTTP** interfészt biztosít a **JAIDE** modellel való interakcióhoz. Kezeli a modell betöltését, a kérések hitelesítését, a sebességkorlátozást (Rate Limiting) és a kötegelt (batch) inference végrehajtást. Végpontok: **POST** /v1/inference a szöveggeneráláshoz és **GET** /health az állapotellenőrzéshez. Biztonság: A RateLimiter biztosítja a rendszer stabilitását, és támogatja az **API** kulcsos hitelesítést.

---

 Formális Verifikáció és Biztonság

A **JAIDE** kódbázis egy többrétegű formális verifikációs és biztonsági infrastruktúrát alkalmaz a matematikai helyesség, a memóriabiztonság és a kriptográfiai integritás biztosítása érdekében.

Lean 4: Mély strukturális bizonyításokhoz használják az **FNDS**, az **RSF** réteg tulajdonságai és a temporális gráf konzisztenciája kapcsán. Agda: Konstruktív bizonyításokat nyújt a memóriabiztonsághoz és az **NSIR** gráf invariánsaihoz. Cryptol & **SAW**: A rendszerkonstansok matematikai specifikációját nyújtja, és validálja a lefordított Zig/**LLVM** kódot. Viper: Biztosítja a heap biztonságot a teljesítménykritikus komponensekhez, mint a Ranker. Circom/ZK: Nem interaktív bizonyításokat generál az inference nyomvonalakhoz Poseidon hashing használatával.

---

 Tesztelés és Fuzzing

A **JAIDE** rendszer többszintű tesztelési stratégiát alkalmaz: Unit Tesztelés: A natív Zig build rendszert használja a specifikus alrendszerek granuláris verifikációjához. Fuzzing: Randomizált bemeneteket biztosít a memóriafoglalási alrendszer (fuzz_memory.zig), a tensor műveletek (fuzz_tensor.zig) és az **SSI** index (fuzz_ssi.zig) teszteléséhez. Stressz Tesztelés: A stress_tensor_refcount.zig segédprogram a referenciaszámláló rendszer atomi műveleteinek telítésére szolgál többszálas környezetben.

---

 Szójegyzék

**KGRU**: Kalmár-Gábor-Riesz Unity, a **JAIDE** architekturális filozófiája. **RSF**: Relational Signal Flow, a mag neurális hálózati architektúra. **MGT**: Morpheme-Guided Tokenization, morféma-alapú tokenizer. **SFD**: Stochastic Fractal Descent, az optimalizációs algoritmus. **SSI**: Succinct Semantic Index, a szemantikus kereső index. **NSIR**: Self-Similar Relational Graph, a tudásreprezentációs gráf. **ESSO**: Entangled Stochastic Symmetry Optimizer, a gráf optimalizálója. Z-Runtime: A relációs műveletek végrehajtási környezete. Chaos Core: A dinamikus feladatütemezést kezelő kernel. **BPE**: Byte Pair Encoding merge szabályok, amelyeket a tokenizer használ. AllReduce: Kollektív kommunikációs művelet a gradiens szinkronizációhoz.
