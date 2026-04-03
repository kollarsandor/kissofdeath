kissofdeath (**JAIDE** v40) – Teljes dokumentáció (Magyar)

> Forrás: [https://deepwiki.com/kollarsandor/kissofdeath](https://deepwiki.com/kollarsandor/kissofdeath) > Generálva: **2026**-04-03

---

 Projektáttekintés

A **JAIDE** (v40) egy gyökérszintű nagy nyelvi modell (**LLM**) rendszer, amelyet a **KGRU** (Kalmár László – Gábor Dénes – Riesz Frigyes egység) architekturális filozofia alapján terveztek. Ez egy vertikálisan integrált stack, amely egyedi neurális architektúrát, dedikált relációs motort, hardveres gyorsítási kerneleket és átfogó formális verifikációs csomagot foglal magában.

A projekt elsősorban Zig nyelven íródott (0.13.0 vagy 0.14.1 verzió), a teljesítménykritikus kernelekhez Futhark nyelvet, a formális specifikációkhoz Lean 4, Cryptol és **SAW** eszközöket használ.

 Rendszer céljai és filozofiája

A **JAIDE** célja, hogy túllépjen a standard transformer architektúrákon az alábbi elemek segítségével:

**RSF** (Relational Signal Flow): Egy olyan neurális rétegkészlet, amely a relációs adatmozgást helyezi előtérbe.

**MGT** (Morpheme-Guided Tokenization): Egy tokenizáló, amely morfológiai struktúrát és morfémabontást alkalmaz a szimpla bájt-pár helyett.

**SFD** (Stochastic Fractal Descent): Egyedi optimalizációs stratégia a paraméterfrissítéshez, pillanatbecslés alapján.

Formális rigor: Minden alapkomponenst matematikai bizonyítékok támasztanak alá a magas kockázatú következtetési feladatokban való biztonság és helyesség garantálása érdekében.

A **KGRU** név magyar úttörőknek állít emléket: Kalmár László (logika/verifikáció), Gábor Dénes (információelmélet/jelterjedés) és Riesz Frigyes (funkcionálanalízis/kvantumlogika).

 Magas szintű architektúra

A rendszer három fő síkra oszlik: az ML Pipeline, a Core Relational Engine és az Acceleration Layer.

Az adatáramlás sorban: nyers szöveg → **MGT** tokenizálás → **RSF** rétegek → **SFD** optimalizáló → **SSI** index → **NSIR** relációs gráf → Ranker visszakeresés.

 Könyvtárstruktúra

| Könyvtár | Cél | Kulcsentitások |
|---|---|---|
| src/core/ | Alapprimitívek | Tensor, Memory, IO |
| src/processor/ | Neurális architektúra | RSF, LayerCore, OFTB |
| src/core_relational/ | Következtetési motor | NSIR, ZRuntime, ReasoningOrchestrator |
| src/hw/ | Hardveres gyorsítás | Futhark, RTL (Clash), CUDA |
| src/verification/ | Formális bizonyítékok | Lean 4, Viper, SAW, Agda |
| src/api/ | Telepítés | InferenceServer |

---

 Bevezetés: Build és konfiguráció

 Eszközlánc és követelmények

A **JAIDE** a Zig programozási nyelven épül, és az **LLVM**-et használja formális verifikációhoz és bitkód generáláshoz.

Alapfüggőségek: Zig 0.13.0, **LLVM** 17, Futhark (**GPU** kernelekhez), **SAW** 0.9.0, Cryptol 2.13.0 és Z3 a formális verifikációs pipeline-hoz.

A src/build.sh szkript automatizálja a szükséges eszközláncok telepítését Linux (apt/dnf/pacman) és macOS (brew) környezetekben.

 Build célok

| Cél neve | Bináris neve | Cél |
|---|---|---|
| jaide | jaide | Standard CPU/GPU interaktív és betanítási mód |
| jaide-gpu | jaide-gpu | Dedikált GPU-optimalizált futtatási útvonal |
| jaide-distributed | jaide-distributed | Több csomópontos betanítás koordináció |
| jaide-inference-server | jaide-inference-server | HTTP API modell kiszolgáláshoz |

Build beállítások: **GPU** gyorsítás -Dgpu=true-val, optimalizálási szintek -Doptimize-zal (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall).

 Működési módok és **CLI** zászlók

Betanítási mód (--mode train): --dataset, --epochs, --batch-size, --learning-rate, --dim / --layers.

Következtetési mód (--mode infer): --model, --prompt, --max-tokens.

 Felhőalapú telepítés (Modal)

A **JAIDE** Python szkriptek átfogó készletét tartalmazza a Modal felhőplatformra való telepítéshez, **NVIDIA** **B200** **GPU**-kat célozva. Az egyedi Ubuntu 22.04 kép **CUDA** 12.4-et, Python 3.11-et és Zig 0.13.0 eszközláncot tartalmaz, csomópontonként akár 8 **B200** **GPU**-val és **256** GB **RAM**-mal.

Telepítési munkafolyamat: (1) modal_setup.sh inicializálja a tokent és köteteket, (2) modal_distributed_train.py futtatja a betanítást, (3) modal_inference.py lehetővé teszi a felhőbeli tesztelést.

---

 Rendszerarchitektúra – Áttekintés

A **JAIDE** (v40) egy nagy teljesítményű **LLM** infrastruktúra, amely hagyományos lineáris szekvenciafeldolgozásról fraktális, relációs gráfalapú megközelítésre vált.

 Magas szintű adatáramlás

## Bevitel: A nyers szöveget az MGT (Morfémavezérelt Tokenizáló) dolgozza fel.

## Feldolgozás: A tokenek embeddingekké alakulnak, majd RSF rétegeken haladnak át. ## Optimalizálás: A gradienseket az SFD optimizer kezeli. ## Indexelés: A feldolgozott jeleket az SSI (Tömör Szemantikai Index) tárolja. ## Relációs logika: A CREVPipeline relációs hármasokat nyom ki az NSIR gráfhoz. ## Visszakeresés: A Ranker LSH-alapú pontozással kéri le a legjelentősebb kontextust.

 Adatreprezentáció

A rendszer egyedi fixpontos típusokat használ a determinisztikus futtatáshoz:

| Típus | Bitszélesség | Skálázási tényező | Felhasználás |
|---|---|---|---|
| FixedPoint16 | 16-bit | 256.0 | Alacsony precizitású súlyok |
| FixedPoint32 | 32-bit | 65536.0 | Standard aktivációk |
| Fixed32_32 | 64-bit | 4294967296.0 | Nagy precizitású gradiensek |

 Alrendszerek összefoglalója

| Alrendszer | Kulcsfájl | Szerepkör |
|---|---|---|
| Processzor | src/processor/rsf.zig | Neurális jelkonverzió |
| Tokenizáló | src/tokenizer/mgt.zig | Szöveg ↔ egész szám leképezés |
| Optimalizáló | src/optimizer/sfd.zig | Fraktálalapú súlyfrissítések |
| Index | src/index/ssi.zig | Tömör szemantikai tárolás |
| Relációs | src/core_relational/nsir_core.zig | Gráfalapú tudáslogika |
| Következtetés | src/api/inference_server.zig | HTTP/REST API kezelés |

---

 Alapprimitívek

Az alapprimitívek biztosítják a szükséges absztrakciókat a többdimenziós tömbök manipulálásához, a speciális memóriakezeléshez és a robusztus modelszerializációhoz.

 Tensor: Többdimenziós tömbmotor

A Tensor struct az src/core/tensor.zig-ben a numerikus számítás elsődleges adatstruktúrája. Főbb jellemzők:

Referenciaszámlálás: Atomi referenciaszámláló (refcount) a memória több tulajdonos közötti kezeléséhez.

Copy-on-Write (CoW): Fizikai másolat csak mutáció esetén készül megosztott pufferről.

Memória integráció: Egyedi allokátorokkal (Arena, Pool, Buddy) kompatibilis.

 Memóriakezelés

| Allokátor | Cél | Szálbiztonság |
|---|---|---|
| Arena | Gyors, tömeges allokációk átmeneti adatokhoz | Mutex védett |
| PoolAllocator | Rögzített méretű blokkallokáció egyforma objektumokhoz | Lock-free / Mutex |
| SlabAllocator | Hatékony allokáció kettő-hatványos méretekhez | Mutex védett |
| BuddyAllocator | Változó méretű allokáció csökkentett töredezettséggel | Mutex védett |

A rendszer szigorúan betartja a secureZeroMemory politikát – érzékeny adatok törlődnek felszabadításkor.

 I/O, típusok és modelszerializáció

Fixpontos aritmetika: FixedPoint16, FixedPoint32, FixedPoint64, Fixed32_32 típusok determinisztikus viselkedéshez.

**PRNG** és BitSet: Xoshiro256++ implementáció sztochasztikus folyamatokhoz; dinamikus bitset állapot követéséhez.

Pontossági szintek: fp4, fp8, fp16, fp32, fp64.

Modelszerializáció: **JAIDE40**\x00 mágikus fejléc, **JSON**-kódolt metaadatok, komponensmágikus számok (**RSF**, **MGT**, **RANKER**, **PROJ**).

Fájl I/O: Szálbiztos **MMAP** absztrakció, stabil mixHash és biztonságos véletlenszám-generálás (Blake2b256 + std.crypto.random).

---

 **LLM** Pipeline – Összefoglalás

Az ML pipeline öt fő komponensből áll a **KGRU** modell betanítási és következtetési logikájához. Az adatáramlás: nyers szöveg → **MGT** tokenizáló → **RSF** processzor → **SFD** optimalizáló → **SSI** index + Ranker.

---

 **RSF** Processzor (Relational Signal Flow)

A **JAIDE** rendszer alapvető neurális hálózati motorja az src/processor/rsf.zig-ben implementált.

 LayerCore struktúra

| Mező | Típus | Leírás |
|---|---|---|
| s_weight | Tensor | Térbeli súlymátrix (dim × dim) |
| t_weight | Tensor | Időbeli súlymátrix (dim × dim) |
| s_bias | Tensor | Térbeli bias vektor (1 × dim) |
| t_bias | Tensor | Időbeli bias vektor (1 × dim) |
| rwlock | Thread.RwLock | Paraméterfrissítések szinkronizálása |

 Inicializálás és validálás

Xavier inicializálás: Súlyok mintavételezése egyenletes eloszlásból a $\pm \sqrt{6/(fan_{in} + fan_{out})}$ határon belül.

Kulcsfüggvények: initOwned, validateTensor2D, ensureFiniteSlice (NaN/Inf ellenőrzés).

 Számítási pipeline

Előre menet: mátrix-szorzás a bemeneten térbeli és időbeli súlyokkal, majd bias hozzáadása.

Visszafelé menet: Gradiens vágás -5.0 és 5.0 között (explodáló gradiens megelőzése), ensureGradients lusta gradiens allokálással.

 Modelmentés/betöltés

Bináris szerializáció verziózással (SAVE_VERSION = 4). A tensorsOverlap függvény biztosítja, hogy forrás és cél pufferek ne legyenek azonosak.

 Formális verifikáció

Lean 4 bizonyítékok: verzióinvariáns, alakaritmetika, memóriabiztonság.

---

 **MGT** Tokenizáló (Morpheme-Guided Tokenization)

Specializált tokenizáló, amely nem csupán bájt-pár kódolást (**BPE**), hanem morfémaalapú bontást is alkalmaz, különösen magyar és angol szövegekhez.

 Alapadatstruktúrák

| Mező | Típus | Leírás |
|---|---|---|
| token_to_id | StringHashMap(u32) | Szöveg → ID leképezés |
| id_to_token | AutoHashMap(u32, []const u8) | ID → szöveg leképezés |
| prefixes | StringHashMap(u32) | Ismert prefixek |
| suffixes | StringHashMap(u32) | Ismert szuffixek |
| roots | StringHashMap(u32) | Alapszótövek |
| anchors | StringHashMap(u64) | SSI anchor tokenek |
| bpe_pairs | StringHashMap(BPEMerge) | Megtanult összevonási párok |

Speciális tokenek: [**PAD**] (0), [**UNK**] (1), [**BOS**] (2), [**EOS**] (3).

 Morfémabontás

Angol prefixek: un, re, pre, dis, mis stb.

Magyar prefixek: meg, el, fel, le, be, ki, szét stb.

Angol szuffixek: ing, ed, tion, ness, ment stb.

Magyar szuffixek: ság, ség, ban, ben, hoz, nak, nek stb.

 Allokátor integráció

initWithArena, initWithPool, initWithBuddy az egyedi allokátorokhoz. Az allocated_strings ArrayList biztosítja a memóriakezelést a leálláskor.

---

 **SFD** Optimalizáló (Stochastic Fractal Descent)

A **KGRU** modell elsődleges betanítási motorja, amely **RSF** rétegparamétereken kezeli a súlyfrissítéseket.

 **SFD** struktúra

SFDConfig: learning_rate, beta1 (momentum bomlás), beta2 (skálázási bomlás), epsilon (numerikus stabilitás).

SFDParam: Egy paraméter súlyát, gradienst és két pillanat tensorát tartalmazza.

 Gradiens frissítési szabály

## Momentum frissítés: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$

## Variancia frissítés: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ ## Bias korrekció az aktuális időlépés alapján ## Súly módosítás: $w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ ## Kvantálás csökkentett pontossági célok esetén

 Kvantálás támogatás

- fp4: Értékek -8.0 és 7.0 közé szorítva, 0.5 pontossággal
- fp8: Értékek -**448**.0 és **448**.0 közé szorítva, 1/16-os pontossággal
- fp16: 1/**1024** pontosság

---

 **SSI** Index (Succinct Semantic Index)

Nagy teljesítményű, trie-alapú hash-fa struktúra token sorozatok indexelésére és visszakeresésére.

 Belső adatstruktúrák

| Struktúra | Szerepkör | Kulcsmezők |
|---|---|---|
| Segment | Összefüggő token sorozat | tokens, position, score, anchor_hash |
| Node | Trie ág vagy levél | hash, children, segment, collision_chain |
| CollisionNode | Láncos lista hash ütközésekhez | seg, next |

Konstansok: Bucket szélesség 6 bit, 64 bucket, max mélység 6 szint.

 Hashelés és indexelés

mixHash: Multiplikatív hash 0x9E3779B185EBCA87 konstanssal. hashTokens: 64-bit hash token sorozathoz. computeAnchorHash: Dokumentum pozíció és token sorozat kombinálásával.

 Sorozat beszúrás (addSequence)

Az anchor_hash kiszámítása után a gyökértől bejárja a fát, levélbe szúr be vagy ütközési lánchoz fűz.

 Hardveres gyorsítás (SSISearch)

Clash-ben írt Mealy állapotgép három állapottal: Idle, Fetching (memóriából csomópontot kér), Comparing (kulcsot hasonlít). Max mélység 64, 32-bit mutatók, 64-bit hash kulcsok.

 Formális verifikáció és fuzz tesztelés

Lean 4 formális specifikáció, **5000** iterációs fuzz tesztelő.

---

 Ranker (**LSH**-alapú eredmény rangsorolás)

A pipeline utolsó szakasza az **SSI** jelöltjeinek pontozásáért és újrarangsorolásáért felelős.

 Konfigurációs konstansok

| Paraméter | Érték | Leírás |
|---|---|---|
| STREAMING_BUFFER_SIZE | 1024 | Max kapacitás a streaming rankerhez |
| DIVERSITY_WEIGHT | 0.3 | Token változatosság hatása |
| PROXIMITY_WEIGHT | 0.3 | Anchor token közelség hatása |
| BASE_SCORE_WEIGHT | 0.4 | Elsődleges n-gram SSI pontszám súlya |
| OVERLAP_WEIGHT | 0.3 | Közvetlen token átfedés súlya |
| JACCARD_WEIGHT | 0.3 | Jaccard hasonlóság súlya |

 Pontozási módszertan

Alappontozás: N-gram analízis decay-súlyozással, sokszínűség-számítás (egyedi/összes token aránya), anchor közelség mérés.

Lekérdezésalapú újrarangsorolás: Token átfedés (lekérdezési tokenek aránya a célsorozatban), Jaccard hasonlóság (intersection/union).

 Formális verifikáció

Viper segítségével verifikálják a halom-biztonságot és matematikai invariánsokat.

---

 Core Relational Engine (Alaprelációs Motor)

A **JAIDE** elsődleges következtetési és tudásreprezentációs alrendszere, amely az **NSIR** gráfot és kvantumlogikát ötvözve nem-lineáris következtetésre képes.

 Rendszerkomponensek

| Komponens | Kód entitás | Elsődleges felelősség |
|---|---|---|
| NSIR Gráf | SelfSimilarRelationalGraph | Tudást fraktális gráfként tárol |
| Kvantumlogika | RelationalQuantumLogic | Kvantum kapukat alkalmaz valószínűségi következtetéshez |
| Z-Runtime | ZRuntime | Relációs műveletek végrehajtási állapotát kezeli |
| ESSO Optimalizáló | EntangledStochasticSymmetryOptimizer | Gráf topológiát optimalizál szimmetria alapján |
| Chaos Core | ChaosCoreKernel | Aszinkron feladatütemezés és adatáramlás |
| Orchestrator | ReasoningOrchestrator | Helyi, globális és meta szintű következtetés koordinálása |

---

 **NSIR** Gráf és Kvantumlogika

 Csomópont és Qubit reprezentáció

| Típus | Mező | Leírás |
|---|---|---|
| Qubit | a | Komplex amplitúdó a |0⟩ bázisállapothoz |
| Qubit | b | Komplex amplitúdó a |1⟩ bázisállapothoz |
| Node | id | Egyedi csomópontazonosító |
| Node | qubit | A csomópont kvantumállapota |
| Node | phase | Fázisszög interferencia-számításokhoz |

 Él minősége

| EdgeQuality | Leírás |
|---|---|
| superposition | Kapcsolat egyszerre több potenciális állapotban létezik |
| entangled | Célcsomópont állapota a forráscsomóponttól függ |
| coherent | Stabil kapcsolat fenntartott fázisszinkronizálással |
| collapsed | Klasszikus, rögzített kapcsolat |
| fractal | Önhasonló kapcsolat, különböző léptékekben ismétlődik |

 Kvantum kapurendszer

Egyes-, kétqubites (**CNOT**) és háromqubites (Toffoli) kapuk, FRACTAL_TRANSFORM doménspecifikus kapu. Elérhető: Hadamard, Pauli-X/Y/Z, Relációs **AND**/OR/**XOR**. A QuantumCircuit struct kapusorozatok végrehajtásához.

 Z-Runtime és végrehajtási előzmények

A HistoryEntry egyedi műveleteket rögzít (assign, transform, measure). Az ExecutionAction magas szintű műveleteket definiál (entangle_variables, propagate_information).

 Jelterjedés

A SignalPropagationEngine az amplitude, phase és frequency alapján szimulálja a jelterjedést. Az EdgeQuality és fractal_dimension alapján transzformálja a jelállapotokat.

 Időbeli gráfkezelés

A TemporalGraph a csomópontok és élek historikus állapotát Timestamp-pel tárolja. A Lean 4 verifikáció biztosítja a kvantumállapot-átmenetek konzisztenciáját.

---

 **ESSO** Optimalizáló és Chaos Core

 **ESSO**: Összefonódó Sztochasztikus Szimmetria Optimalizáló

Szimmetria-transzformációk az **NSIR** gráf optimalizálásához: Identity, Reflection, Rotation (90/**180**/**270**), Translation. Az OptimizationState követi az aktuális gráfot, energiáját és szimmetria-előzményeit.

Kulcsfüggvények: optimize() (sztochasztikus leszállás), applySymmetryToGraph() (transzformáció alkalmazása), calculateGraphEnergy() (energia értékelése).

 Chaos Core

Tartalom-Archiváló Tárolás (**CAS**): A MemoryBlock entitásokat tartalomhash azonosítja, lehetővé téve a deduplikálást és a függőségnyomkövetést.

Dinamikus feladatütemező: A DynamicTaskScheduler prioritás és adatfüggőségek alapján rendel feladatot magokhoz.

| Komponens | Felelősség |
|---|---|
| ContentAddressableStorage | Tartalom-archiváló memóriakezelés |
| DynamicTaskScheduler | Függőségtudatos feladat-végrehajtás |
| DataFlowAnalyzer | Adatáramlás és szűk keresztmetszetek elemzése |
| ChaosCoreKernel | Alkomponensek orkestrálása |

 Meglepetés Memória

Prioritás-alapú adatmegőrzés újdonság szerint. SurpriseMetrics három tényező alapján: Jaccard diszhasonlóság, tartalomhash-távolság, időbeli újdonság.

Konfigurációs konstansok: RETENTION_BASE_WEIGHT = 0.5, RETENTION_AGE_WEIGHT = 0.3, RETENTION_FREQUENCY_WEIGHT = 0.2, DEFAULT_SURPRISE_THRESHOLD = 0.3.

---

 Következtetési Orkesztrátor és Támogató Modulok

 Következtetési Orkesztrátor

Háromszintű következtetés: Helyi (azonnali csomópontszomszédságok), Globális (hosszú hatótávolságú strukturális igazítás), Meta (a következtetési folyamat önreflexiója).

 Relációs **GPU** (R-**GPU**) és NoC

Szimulált hardverabsztrakció masszívan párhuzamos gráfműveletekhez. Az AsynchronousNoC üzenetirányítást kezel prioritás-alapú sorral. Üzenettípusok: weight_update, graph_sync, isomorphism_result.

 Vektorprocesszor (**VPU**)

**SIMD** absztrakciók: aritmetika (add, sub, mul, divChecked), geometria (dot, magnitude, normalize), hardveres gyorsítás (fma, sqrt).

 **CREV** Pipeline

Feldolgozási szakaszok: tokenizálás → hármas kinyerés (Subject-Relation-Object) → validálás → integráció az **NSIR** gráfba → **SSI** indexelés.

 Adathalmazok elfedése és biztonság

Paillier Homomorphic Encryption: Matematikai műveletek titkosított szövegen végrehajthatók. encrypt(plaintext: i64) → u512 ciphertext, add(c1, c2) → homomorphic összeadás.

Biztonság: safeIntCast, safePtrCast (határellenőrzés), secureZeroBytes (volatile írások), secureCompare (konstans idejű összehasonlítás).

 C **API**

Stabil **FFI**: CGraph ↔ GraphContext, COptimizer ↔ EntangledStochasticSymmetryOptimizer. Hibakódok: JAIDE_SUCCESS (0), JAIDE_ERROR_NULL_POINTER (-1), JAIDE_ERROR_OUT_OF_MEMORY (-18).

---

 Kvantumszámítási Integráció

 Architektúra áttekintés

Három fő komponens: Hardware Abstraction (quantum_hardware.zig), Cloud Integration (ibm_quantum.zig), Task Adaptation (quantum_task_adapter.zig).

 **IBM** Quantum integráció

initWithCrn: **API** tokennel és **CRN**-nel inicializál. submitJob: OpenQASM küldés ibm_brisbane backendre, **1024** shottal. getJobResult: Állapot és mérési adatok lekérése.

 Hardver specifikációk

| Backend | Qubitek | T1 átlag (ns) | T2 átlag (ns) | Readout hiba |
|---|---|---|---|---|
| Heron | 133 | 350,000 | 200,000 | 0.008 |
| Eagle | 127 | 200,000 | 120,000 | 0.015 |
| Falcon | 27 | 100,000 | 80,000 | 0.020 |
| Condor | 1121 | 400,000 | 250,000 | 0.006 |

 Kvantum feladatadapter

Magas entanglement és fraktális dimenzióval rendelkező részgráfokat azonosít. Entanglement_threshold alapértelmezetten 0.5, fractal_threshold alapértelmezetten 1.5.

Konfigurációs konstansok: MAX_QUBITS_SIMULATION = 20, SIMULATOR_MAX_SHOTS = **100**,**000**, JOB_WAIT_TIMEOUT_MS = 60,**000**.

---

 Hardveres Gyorsítás

A hardveres gyorsítás egy többszintű stack a **GPU**, **FPGA** és **ASIC** célokra, feltételesen engedélyezhető a gpu_acceleration build jelzővel.

---

 Futhark Gyorsítási Réteg

 Rendszeráttekintés

Három réteg: Futhark Kernelyek (.fut fájlokban), Zig **FFI** Kötések, Acceleráció Interfész (magas szintű Zig wrapper).

 Futhark Kernelyek

**RSF** Forward and Backward: Spektrális transzformáció (weights_s-szel), időbeli transzformáció (weights_t-vel), fúzionált forward (ReLU + Layer Normalization egyetlen **GPU** hívásban).

**SFD** optimalizálás: Momentum-alapú súlyfrissítés **GPU**-n: $v_{t+1} = \mu v_t + \eta g$, $w_{t+1} = w_t - v_{t+1}$.

 Zig Acceleráció Interfész

FutharkContext: **GPU** eszköz életciklus és alapértelmezett hangolási paraméterek kezelése.

PinnedMemory: cudaHostAlloc-ot használ gyors **DMA**-átvitelhez, cudaFreeHost-tal felszabadítva.

FutharkArray2DF16: **GPU** tömb életciklus, futhark_new_f32_2d, futhark_values_f32_2d, futhark_free_f32_2d.

 Fractal **LPU**

A FractalTile struktúra rekurzívan osztja a memóriát hausdorff_dim (Hausdorff Dimenzió) alapján. A balanceLoad normalizálja a pending_ops értékeket a számítási egységek között.

---

 **RTL** Hardvermodulok (Clash/Haskell)

 SSISearch: Bináris fa keresőmotor

Adattípusok: HashKey64 (64-bit hash), NodeAddr32 (32-bit memóriacím), TreeNode (nodeKey, leftChild, rightChild, érvényességi bit).

Keresési állapotgép:

| Állapot | Leírás |
|---|---|
| Idle | SearchRequest-re vár |
| Fetching | Memóriavezérlőtől TreeNode adatot vár |
| Comparing | searchKey-t hasonlít a nodeKey-hez |

Keresési logika: egyezés → found = True; kisebb → leftChild fetch; nagyobb → rightChild fetch; max mélység (64) → befejezés.

 RankerCore: Pontozási pipeline

finalScore = baseScore + positionBias, ahol positionBias = positionBiasScale / (position + 1) és positionBiasScale = **1000**.

 MemoryArbiter: Többklienses busz-arbitrátor

Max 4 kliens, round-robin politika, ServiceCycles = 4. A filterResp biztosítja, hogy csak a kérő kliens kapja meg az adatát.

---

 **FPGA** Implementáció

 Rendszerarchitektúra

A top_level.v **FPGA** felső szintű modul integrálja: **AXI4**-Lite Slave (vezérlési interfész), SSISearch Core, RankerCore, MemoryArbiter.

 **AXI4**-Lite regisztertérkép

| Cím | Regiszternév | Leírás |
|---|---|---|
| 16'h0000 | ADDR_CONTROL | 0. bit: SSI Start; 1. bit: Ranker Valid |
| 16'h0004 | ADDR_STATUS | 0. bit: SSI Found; 1. bit: Ranker Done; 16-31 bitek: Rank |
| 16'h0010 | ADDR_SSI_KEY_L | SSI Keresőkulcs (alsó 32 bit) |
| 16'h0014 | ADDR_SSI_KEY_H | SSI Keresőkulcs (felső 32 bit) |
| 16'h0018 | ADDR_SSI_ROOT | SSI gyökércsomópont memória-cím |
| 16'h0038 | ADDR_RNK_SCORE | Ranker kalkuláció alap pontszáma |
| 16'h003C | ADDR_RNK_RES | Ranker végső számított pontszáma |

 Fizikai megszorítások és időzítés

Órajel: **100** MHz (J3 láb). Reset: aktív-alacsony, rst_n (**K11** láb).

Többciklusú útvonalak: Arbiter → Memory: 4 ciklus setup; **SSI** → Memory: 8 ciklus setup.

---

 **ASIC** Implementáció (**TSMC** 28nm)

 Szintézis folyamat

Synopsys Design Compiler, **TSMC** 28nm standard cell könyvtárak (slow.db, typical.db, fast.db). Órajel periódusa 10.0 ns (**100** MHz), uncertainty 0.2 ns.

Többciklusú útvonalak: MemoryArbiter: 4× setup, 3× hold; SSISearch: 32× setup, 31× hold.

Optimalizálás: compile_ultra -gate_clock (clock gating), területi cél = 0 (legkisebb lábnyom).

 Floorplanning és fizikai tervezés

Die mérete **5000**×**5000** egység, core offset **100** egység, sor-arány 0.70.

PG háló: **METAL6** (vízszintes) és **METAL5** (függőleges), 10.0 szélesség, PG strap pitch **120**.0.

Pin elhelyezés:

| Jelcsoport | Die oldal | Rétegek |
|---|---|---|
| AXI Write | Bottom | METAL5, METAL6 |
| AXI Read | Right | METAL5, METAL6 |
| Memory | Left | METAL5, METAL6 |
| System (clk, rst_n) | Left (Offset) | METAL6 |
| Peripherals | Top | METAL5, METAL6 |

Kimeneti fájlok: top_level_floorplan.v, top_level_floorplan.def, top_level_floorplan.tcl.

---

 Elosztott Betanítás

 Rendszeráttekintés

Hierarchia: **GPU** Koordinátor (eszközspecifikus erőforrások) → Elosztott Trainer (magas szintű ciklus, particionálás, szinkronizálás) → Futhark-gyorsított Trainer (**GPU** kernelek).

 **GPU** Koordinátor

Eszközkezelés (cudaSetDevice), **NCCL** kommunikátorok inicializálása, memória műveletek (cudaMalloc, cudaFree, cudaMemcpy), allReduce és broadcast float32/float16 típusokhoz. Barrier mechanizmus dummy allReduce hívással szinkronizáláshoz.

 Elosztott Trainer (Futhark)

Adathalmaz-szeletelés: sorokat számol a .jsonl adathalmazban, base_samples_per_rank kiszámítása, minden rank a kijelölt start_line-tól olvas.

Gradiens aggregáció: helyi gradiens számítás, majd GPUCoordinator.allReduceFloat16 hívás a world_size-szal való osztás előtt.

 Tensor primitívek elosztott betanításhoz

N-dimenziós alakok stride-alapú indexeléssel, atomi referenciaszámlálás, kontiguus memória ellenőrzés az **NCCL** átvitelekhez. Fixed32_32 típus determinisztikus ellenőrzéshez.

 Formális verifikáció

Lean 4 bizonyítja a Fixed32_32 kommutatív és asszociatív törvényeit, a **PRNG** determinizmusát és a határon belüli maradást.

---

 **NCCL** Kötések és Modal Felhő Telepítés

 **NCCL** Kötések

| Függvény | Leírás |
|---|---|
| ncclGetUniqueId | Egyedi azonosítót generál a bootstrap kommunikátorhoz |
| ncclCommInitRank | Új kommunikátor objektumot hoz létre egy adott rankhoz |
| ncclAllReduce | Redukciót végez az összes GPU-n és elosztja az eredményt |
| ncclBroadcast | Másolja a puffert az összes rankra |
| ncclReduceScatter | Redukciót végez és szétszórja az eredményt |

**CUDA** helper integrációk: cudaMalloc, cudaFree, cudaMemcpy, cudaStreamCreate, cudaStreamSynchronize, cudaSetDevice, cudaGetDeviceCount.

 Modal **GPU** Absztrakció

ModalGPUClient Zig-natív interfész a Modal Cloud **API**-hoz. Inicializáláskor 8 **GPU**-t céloz (**B200**). deployTrainingJob: **JSON** payload a jaide-v40-training képpel, getJobStatus: feladatállapot lekérés.

 Felhő Telepítési Szkriptek

Infrastruktúra beállítás (modal_setup.sh): Modal **CLI** telepítés, hitelesítés, jaide-training-data kötet létrehozása, **API** kulcstitkosítások.

Elosztott betanítás (modal_distributed_train.py): **B200**:8 + **256** GB **RAM** + 3 TB lemez, Ubuntu 22.04 + **CUDA** 12.4 + Zig 0.13.0, HunSum-1 adathalmaz letöltés és **JSONL** konverzió, 8 **GPU**-n elosztott betanítás.

Következtetési szkript (modal_inference.py): **B200**:1 **GPU**, kötegelt feldolgozás prompt-lista iterálásával, **CLI** belépési pont modal run-nal.

---

 Következtetési Szerver **API**

 Szerverfiguráció

| Mező | Típus | Alapértelmezett | Leírás |
|---|---|---|---|
| port | u16 | 8080 | TCP port |
| host | []const u8 | *127.0.0.1* | Kötési cím |
| max_connections | u32 | 100 | Max egyidejű kapcsolatok |
| batch_size | usize | 32 | Kérések száma következtetési menetenként |
| rate_limit_per_minute | u32 | 10 | Max kérés IP-nként percenként |
| require_api_key | bool | true | API kulcs validálás engedélyezése |

 Kéréskezelési pipeline

## Kapcsolat fogadás a konfigurált gazdagépen/porton

## Sebességkorlátozás ellenőrzése (csúszóablak, 60 másodperces ablak) ## JSON elemzés InferenceRequest struct-okká ## Tokenizálás MGT segítségével ## Forward pass az RSF réteg stackon ## JSON szerializálás InferenceResponse-ban

 **API** Végpontok

**POST** /v1/inference: Kérés: text, max_tokens (opcionális), return_embeddings (opcionális). Válasz: tokens, embeddings (opcionális), processing_time_ms.

**GET** /health: Válasz: status (*healthy*), uptime_seconds, model_loaded.

---

 Formális Verifikáció és Biztonság

 Verifikációs Infrastruktúra

| Eszköz | Felhasználás |
|---|---|
| Lean 4 | NSIR, RSF rétegek, temporal gráf, Surprise Memory mély strukturális bizonyításai |
| Agda | Memóriabiztonság, arena allokáció, NSIR gráf invariánsok konstruktív bizonyításai |
| Cryptol & SAW | Rendszerkonstansok specifikációja, Zig/LLVM kód validálása |
| Viper | Ranker halom-biztonság és memória invariánsok |
| Circom/ZK | Nem-interaktív bizonyítékok következtetési nyomokhoz Poseidon hasheléssel |
| Beluga/Mizar/TwElf | RSF réteg relációs logika és típuselméleti alapok |

 Biztonsági és Invariáns Taxonómia

| Invariáns típus | Prioritás | Leírás |
|---|---|---|
| MEMORY_SAFETY | 10 | Puffertúlcsordulások és use-after-free hiánya |
| TYPE_SAFETY | 9 | Típusinvariánsok megőrzése a Z-runtime-ban |
| CONNECTIVITY | 8 | Az NSIR gráf strukturális integritása |
| QUANTUM_STATE | 5 | Érvényes valószínűségeloszlások a kvantumlogika rétegekben |

 Lean 4 Formális Bizonyítékok

Fő bizonyítási területek: **FNDS** helyesség (önhasonlóság megőrzése), **RSF** réteg invariánsok (jeltartósság mély rétegkészleteken), Surprise Memory (retention_priority számítás és exceedsThreshold logika verifikálása).

 **SAW**, Cryptol, Viper, **ACL2**, Agda és ZK Verifikáció

**SAW** és Cryptol: A MainSpec.cry a rendszerhatárok forrása (pl. MAX_TENSOR_SIZE, FILE_MAGIC_RSF). Az src/verify.saw **LLVM** szinten verifikálja a Zig kódot.

Agda memóriabiztonság: sizeConservation invariáns bizonyítása: arenaAllocated + arenaRemaining ≡ bufferSize.

Zero-Knowledge következtetés: inference_trace.circom Poseidon hasheket használ a belső tensor értékek felfedése nélküli verifikációhoz.

**ACL2** és relációs logika: **MGT** tokenizáló állapotátmenetek modellezése. Beluga/Mizar/TwElf az **RSF** referenciaszámlálás és tensor érvényesség bizonyításához.

 Biztonsági Politika

Bell-LaPadula és Biba modellek alapján kötelező hozzáférés-szabályozás. Biztonsági szintek: **PUBLIC**-tól TOP_SECRET-ig. Integritási szintek: **UNTRUSTED**-tól **KERNEL**-ig.

| Biztonsági funkció | Implementáció |
|---|---|
| Hozzáférés-szabályozás | Bitmaszk alapú AccessRight (READ, WRITE, EXECUTE, ADMIN) |
| Információáramlás | dominates ellenőrzés több-szintű biztonsághoz |
| Integritás-ellenőrzések | Időzítés-biztos egyenlőség és többhash-támogatás |

---

 Tesztelés és Fuzzing

 Egységtesztelési Infrastruktúra

| Parancs | Cél forrás | Leírás |
|---|---|---|
| zig build test | src/main.zig | Elsődleges egységteszt csomag (Futhark kernel linkekkel) |
| zig build test-tensor | src/core/tensor.zig | Elszigetelt tesztek a Tensor motorhoz |
| zig build test-memory | src/core/memory.zig | Egyedi allokátorok verifikálása |

 Fuzz Tesztelési Keretrendszer

Memória rendszer fuzzing (fuzz_memory.zig): Véletlenszerű alignedAlloc, free és realloc műveletek. Aktív allokációk nyomon követése, memóriafolyás-ellenőrzés a futás végén.

Tensor műveleti fuzzing (fuzz_tensor.zig): Véletlenszerű alakok (max rank 4), random f32 adatok. Tesztelt műveletek: összegzés, max, L2-norma, skálázás. NaN/Inf ellenőrzés numerikus instabilitásra.

**SSI** index fuzzing (fuzz_ssi.zig): **5000** iteráció, váltakozó addSequence (max **1024** token) és retrieveTopK hívásokkal. Sikeres és sikertelen műveletek, indexelt tokenek összeszámlálása.

 Stressz-tesztelés: Szálbiztonság és Referenciaszámlálás

12 szál egyidejűleg manipulál megosztott Tensor objektumkészletet. Atomi barrier biztosítja az egyidejű start-ot. Tesztelt műveletek: retain, release, clone, set, tömeges aritmetika.

---

 Szójegyzék

 Alaparchitekturális Fogalmak

| Fogalom | Definíció | Implementáció |
|---|---|---|
| KGRU | Kalmár-Gábor-Riesz Unity: a JAIDE alapfilozófiája három magyar matematikusról | README.md |
| RSF | Relational Signal Flow: egyedi neurális architektúra térbeli és időbeli súlymátrixokkal | src/processor/rsf.zig |
| MGT | Morpheme-Guided Tokenization: morfémabontást alkalmazó tokenizáló | src/tokenizer/mgt.zig |
| SFD | Stochastic Fractal Descent: egyedi optimalizáló fp4-től fp64-ig | src/optimizer/sfd.zig |
| Tensor | Többdimenziós tömb referenciaszámlálással és CoW szemantikával | src/core/tensor.zig |
| SSI | Succinct Semantic Index: trie/hash-fa token sorozatokhoz | src/index/ssi.zig |
| NSIR | Self-Similar Relational Graph: kvantumtulajdonságokkal rendelkező gráf | src/core_relational/nsir_core.zig |
| R-GPU | Relational Graph Processing Unit: aszinkron NoC-alapú hardver absztrakció | src/core_relational/r_gpu.zig |
| Anchor Token | Stabil token SSI-beli szegmensek indexeléséhez | src/index/ssi.zig |
| LSH | Locality Sensitive Hashing: a Ranker gyors hasonlóságkereséshez | src/main.zig |
| Z-Runtime | Relációs műveletek és változóállapot kezelési végrehajtási környezete | src/core_relational/mod.zig |
| Chaos Core | Dinamikus feladatütemezés és CAS kezelő kernel | src/core_relational/chaos_core.zig |
| ESSO | Entangled Stochastic Symmetry Optimizer: szimmetriaalapú gráf optimalizáló | src/core_relational/esso_optimizer.zig |
| Fixed32_32 | Egyedi 64-bit fixpontos típus (32+32 bit) precíziókontrolált aritmetikához | src/core/types.zig |
| Lean 4 | RSF helyesség és SSI invariánsok matematikai bizonyításához | src/verification/lean4/ |
| SAW | Software Analysis Workbench: Zig/LLVM bitkód verifikáció | src/verify.saw |
| Cryptol | Rendszerkonstansok magas szintű specifikációja | src/MainSpec.cry |
| ZK | Zero-Knowledge: következtetési nyomok verifikációja súlyok felfedése nélkül | src/zk/inference_trace.circom |
