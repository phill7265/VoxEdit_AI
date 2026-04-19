# 📽️ VoxEdit_AI: Harness Engineering 기반 AI 영상 편집 파이프라인

**VoxEdit_AI**는 AI 모델에만 의존하는 방식을 넘어, 정교한 **하네스(Harness)**와 전문화된 **에이전트 스킬(Agent Skills)**을 결합하여 프로페셔널한 영상 결과물을 산출하는 엔지니어링 엔진입니다.

---

## 🏗️ 핵심 설계 철학: Agent = Model + Harness

[cite_start]본 프로젝트는 "AI의 지능은 모델에서 나오지만, 결과물의 신뢰성은 하네스에서 결정된다"는 원칙을 따릅니다. [cite: 51, 60]

### 1. `/spec` : 진실의 원천 (Source of Truth)
추상적인 지시를 수치화된 규칙으로 변환하여 AI가 일관된 품질을 유지하도록 강제하는 '프로젝트 헌법' 계층입니다.
* [cite_start]**명세 기반 개발**: 0.5초 이상의 무음 제거, 1.1x 점프 컷 확대 등 구체적인 수치를 명세화합니다. [cite: 107, 108]
* **품질 게이트**: 해상도, 오디오 클리핑 등을 자동 검증하는 기술적 표준을 정의합니다. [cite: 89, 90]

### 2. `/harness` : 제어 및 관측성 (Control & Observability)
AI 모델의 변덕을 통제하고 작업의 연속성을 보장하는 실행 인프라입니다. [cite: 51]
* [cite_start]**인수인계 노트 (Handover Notes)**: 세션이 끊겨도 이전 작업 상태를 완벽히 복원하는 `MemoryManager`를 운영합니다. [cite: 79, 80]
* [cite_start]**컨텍스트 방화벽 (Context Firewall)**: **컨텍스트 부패(Context Rot)** 현상을 막기 위해, 각 스킬에 필요한 최소한의 데이터만 선별적으로 주입합니다. [cite: 73, 85, 87]
* **안전한 실험실 (Sandbox)**: FFmpeg 등 외부 명령어를 격리된 환경에서 검증한 후 실행하여 시스템 안정성을 확보합니다. [cite: 81, 82]

### 3. `/skills` : 무상태 전문 모듈 (Stateless Worker Skills)
각 편집 단계에 특화된 기능을 수행하는 독립적인 전문 역량 단위입니다. [cite: 50, 85]
* **무상태성(Stateless)**: 각 스킬은 상태를 직접 소유하지 않으며, 오직 하네스 메모리와 통신하여 결합도를 최소화합니다.
* **전문화된 워크플로**: 
    * **Transcriber**: 음성 파형 분석 및 단어 단위 타임스탬프 추출.
    * **Cutter**: 명세에 기반한 정밀한 컷 리스트(EDL) 설계.
    * **Designer**: 레이아웃 가이드라인을 준수하는 자막 및 시각 효과 배치.
    * **Exporter**: 하네스 샌드박스 내에서 복합 필터 그래프를 통한 최종 렌더링.

---

## 🚀 엔지니어링 기대 가치
* [cite_start]**결정론적 품질**: 모델 업데이트와 무관하게 명세(Spec)에 정의된 기준을 준수합니다. [cite: 63]
* [cite_start]**성능 최적화**: 정교한 하네스 설계만으로 AI의 작업 수행 능력을 극대화합니다. [cite: 43]
* **확장성**: 새로운 편집 스타일이나 검증 로직을 `/spec` 추가만으로 즉시 적용 가능합니다. [cite: 148]

---
*VoxEdit_AI is built on the principles of **Harness Engineering** to turn AI from a conversational partner into a robust production line.*
