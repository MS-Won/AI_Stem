# MS Audio 0단계 (브랜치 정리 + 운영 체계) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 0 브랜치를 main에 합치고, 역할별 에이전트·대표 보고서·알림으로 이루어진 MS Audio 운영 체계를 하루 안에 갖춘다.

**Architecture:** 코드가 아니라 저장소 안 문서와 Claude Code 설정이다. 부서는 `.claude/agents/*.md` 에이전트, 보고와 결정은 `docs/company/` 아래 마크다운, 세션 밖 알림은 클라우드 예약 작업 1개. 메인 세션이 기획실 역할을 한다는 규칙은 `CLAUDE.md`에 둔다.

**Tech Stack:** Claude Code 에이전트 정의(마크다운 + YAML frontmatter), Claude Code 클라우드 예약 작업(`/schedule` 스킬), git/GitHub.

**Spec:** `docs/superpowers/specs/2026-11-26-ms-audio-org-design.md` (조직 §5, 흐름 §6, 알림 §7, 0단계 §8)

## Global Constraints

- 모든 문서·에이전트 설명·보고서·커밋 메시지는 한국어. 커밋 메시지는 왜 그렇게 했는지를 남긴다.
- 0단계 전체는 하루 안에 끝낸다 (반목표 X1). 문구 다듬기로 시간을 쓰지 않는다.
- 대표 보고서의 '요약'+'대표 결정 필요' 두 칸은 합쳐 10줄 이내 (X2).
- 감사실(`auditor`)과 품질팀(`qa`)은 쓰기 도구(Write, Edit, NotebookEdit)를 갖지 않는다. `auditor`는 Bash도 없다.
- 감사 의견은 목표(첫 1원, 출시 기간)에서 벗어난 점을 구체적으로 지적한다. 지적할 것이 없으면 "지적 없음"과 그 근거를 쓴다 (X3).
- 알림 경로는 휴대폰 Claude 앱 푸시 (대표 결정). 예약 작업은 저장소에 아무것도 쓰지 않는다. 결정 대기도 할 일도 없으면 알리지 않는다.
- `Test AR/`, `engine/out/`, `engine/models/`는 커밋하지 않는다.
- 외부 도구(예약 작업, 에이전트 frontmatter)의 형식은 계획서를 믿지 말고 설치된 Claude Code 문서/스킬로 확인한다.

## Review Focus

1. 다른 PC에서 세션을 열 때: 에이전트·규칙이 저장소에 커밋돼 있어 그대로 동작해야 한다 → Task 2·3에서 `.claude/agents/`와 `CLAUDE.md`가 git에 들어갔는지 `git ls-files`로 확인.
2. 비공개 저장소: 예약 작업이 `MS-Won/AI_Stem`을 못 읽으면 알림이 조용히 멈춘다 → Task 4에서 1회 즉시 실행해 실제로 STATE.md 내용이 담긴 푸시가 오는지 확인.
3. 결정 대기·할 일이 없는 날: 빈 알림이 오면 알림 피로 → Task 4 프롬프트에 "없으면 알리지 않는다"를 넣고 문구로 확인.
4. push 안 된 상태: 예약 작업은 GitHub의 main을 읽으므로 세션 끝에 push가 안 되면 어제 상태를 알린다 → Task 2의 CLAUDE.md 규칙에 "보고서 커밋 후 push"를 넣는다.
5. 감사실이 칭찬만 하는 경우 → Task 5에서 실제 감사 실행 결과에 구체적 지적 또는 근거 있는 "지적 없음"이 있는지 확인.

---

### Task 1: Phase 0 브랜치를 main에 합치기

**Files:** 없음 (git 작업)

- [ ] **Step 1: 엔진 테스트 실행**

Run (`engine/`에서): `.venv/Scripts/python -m pytest -m "not gpu" -q`
Expected: 전부 passed (현재 58 passed 수준), failed 0

- [ ] **Step 2: main에 병합**

```bash
git checkout main && git pull --ff-only origin main
git merge --no-ff phase0-engine -m "Phase 0 엔진을 main에 합침 ..."
```
병합 커밋 메시지에는 "Phase 0 종료: 스튜디오 곡 청취 만족, 라이브 끊김은 연구 과제로" 취지를 한국어로 남긴다. 충돌이 나면 멈추고 기획실에 보고한다.

- [ ] **Step 3: 병합 후 테스트 재실행**

Run: Step 1과 같음. Expected: 같은 결과

- [ ] **Step 4: push (대표 확인 후)**

Run: `git push origin main`. 이후 작업은 main에서 이어 간다. `phase0-engine` 원격 브랜치는 지우지 않는다.

---

### Task 2: `docs/company/` 문서 체계와 기획실 규칙

**Files:**
- Create: `docs/company/README.md` — 조직표(스펙 §5 표 그대로)와 일하는 흐름(§6 1~5번) 요약, 30줄 이내
- Create: `docs/company/reports/_template.md` — 스펙 §6의 보고서 형식 그대로
- Create: `docs/company/decisions.md` — 한 줄 형식 `- YYYY-MM-DD 결정 — 이유`. 초기 항목: 스펙 §10의 3건 + 2026-11-26 "라이브 끊김은 연구 과제로 이관 — 숫자 지표는 좋아졌지만 청취상 미해결" + 2026-11-24 "결제는 Phase 1.5" (제품 설계 G5)
- Create: `docs/company/research/README.md`, `docs/company/design/README.md` — 각 1~3줄: 무엇을 두는 곳인지, 파일 이름 규칙 `YYYY-MM-DD-<주제>.md`
- Modify: `CLAUDE.md` — 아래 두 곳
- Modify: `docs/STATE.md` — 맨 위에 `## 대표 확인` 블록 (최신 보고서 링크, 대기 중인 결정 목록)

**Interfaces:**
- Produces: 보고서 경로 규칙 `docs/company/reports/YYYY-MM-DD-<주제>.md`, STATE.md 맨 위 `## 대표 확인` 블록 (Task 4 예약 작업과 Task 5가 이 이름으로 읽는다)

- [ ] **Step 1: 위 문서 파일 작성**

- [ ] **Step 2: `CLAUDE.md` 수정**
  - '세션 시작 시'에 추가: STATE.md의 `## 대표 확인`을 읽고 대기 중인 결정부터 대표에게 묻는다. 메인 세션은 기획실이다 (`docs/company/README.md`).
  - '작업 규칙'에 추가: 큰 작업 단위가 끝나면 `auditor` 점검 → 대표 보고서 작성 → `decisions.md`·STATE.md·todo.md 갱신 → 커밋 → **push** (예약 알림이 GitHub을 읽기 때문). 오래 걸리는 작업이 끝나거나 대표 결정이 필요해지면 푸시 알림을 보낸다.
  - '문서 지도'에 `docs/company/README.md`, 운영 체계 설계 경로 추가.

- [ ] **Step 3: 확인**

Run: `git status --short docs/company CLAUDE.md docs/STATE.md`
Expected: 위 파일이 모두 보임 (무시 규칙에 걸리지 않음)

- [ ] **Step 4: Commit** — `docs/company/`, `CLAUDE.md`, `docs/STATE.md`

---

### Task 3: 부서 에이전트 5개

**Files:**
- Create: `.claude/agents/design.md`, `developer.md`, `research.md`, `qa.md`, `auditor.md`

각 파일: YAML frontmatter(`name`, `description`, `tools`, 필요 시 `model`) + 본문(역할, 먼저 읽을 문서, 산출물 형식과 위치, 하면 안 되는 것). 내용은 스펙 §5 표의 해당 행을 그대로 옮기고 아래를 더한다. frontmatter 필드 이름과 형식은 먼저 Claude Code 문서로 확인한다 (claude-code-guide 에이전트에 질문).

| 에이전트 | 먼저 읽을 것 | tools | 더할 것 |
|---|---|---|---|
| design | 제품 설계 §1~2·§6, `docs/company/design/` | Read, Glob, Grep, Write, Edit, Skill | `frontend-design` 스킬 사용. 시안은 대표 확인용 Artifact로 게시 |
| developer | `CLAUDE.md` 함정, 해당 계획서 Task | 전체 | TDD, pytest/Vitest 통과 후에만 커밋. 계획서 밖 기능 금지 |
| research | `docs/STATE.md` 연구 관련 절, `todo.md` 연구 과제, `docs/company/research/` | 전체 | 지시받을 때 정한 기간 안에서만. 결과는 청취 파일 경로 + 대표가 들을 시각을 포함. 엔진 코드 변경 금지 |
| qa | 해당 계획서와 diff | Read, Glob, Grep, Bash, Skill | 고치지 않고 '통과/문제 목록(파일:줄, 재현)'만 보고 |
| auditor | 두 설계 문서의 목표·반목표·기간, `decisions.md`, 최근 보고서·git log | Read, Glob, Grep | 점검 질문 4개 고정: ①이번 일이 첫 1원/출시에 얼마나 가까워지게 했나 ②기간 대비 진척 ③완벽주의·재미만 쫓기·범위 확장 신호 ④대표가 다음에 하지 말아야 할 것. 각 답은 근거(파일·커밋) 포함 |

- [ ] **Step 1: frontmatter 형식 확인** (claude-code-guide에게: 프로젝트 서브에이전트 파일 위치·필드·tools 쓰는 법)
- [ ] **Step 2: 5개 파일 작성**
- [ ] **Step 3: 확인**

Run: `git ls-files --others --exclude-standard .claude/agents`
Expected: 5개 파일 모두 나옴. 그리고 `qa.md`, `auditor.md`의 tools 줄에 Write/Edit가 없음 (`grep -n "^tools" .claude/agents/*.md`)

- [ ] **Step 4: Commit** — `.claude/agents/`

---

### Task 4: 평일 아침 푸시 알림

**Files:** 없음 (클라우드 예약 작업)

- [ ] **Step 1: `/schedule` 스킬로 예약 작업 생성**
  - 일정: 평일(월~금) 오전 9시 KST
  - 저장소: `MS-Won/AI_Stem` main, 읽기만
  - 가벼운 모델
  - 프롬프트 요지: `docs/STATE.md`의 `## 대표 확인`과 가장 최근 `docs/company/reports/*.md`를 읽고, "결정 대기 N건 / 오늘 할 일 1~3개"를 3줄 이내 한국어 푸시 알림으로 보낸다. 결정 대기와 할 일이 모두 없으면 알리지 않는다. 저장소에 아무것도 쓰지 않는다.
- [ ] **Step 2: 1회 즉시 실행**
Expected: 휴대폰에 STATE.md 내용이 반영된 알림 도착 (대표 확인). 저장소를 못 읽으면 GitHub 연결 후 재실행.
- [ ] **Step 3:** 예약 작업 이름·일정을 `docs/company/README.md`의 '알림' 줄에 기록하고 커밋

---

### Task 5: 첫 감사와 첫 대표 보고서

**Files:**
- Create: `docs/company/reports/2026-11-26-stage0.md` (템플릿 형식)
- Modify: `docs/STATE.md`(`## 대표 확인` 갱신, 단계 줄 'Phase 1 1단계 대기'), `todo.md`(Phase 1 단계 0~5 체크리스트 추가, 0단계 체크), `docs/company/decisions.md`(필요 시)

- [ ] **Step 1: `auditor` 에이전트 실행** — 대상: Phase 0 전체와 0단계. 결과를 보고서 '감사 의견' 칸에 그대로 붙인다.
- [ ] **Step 2: 보고서 작성** — 다음 할 일: 1단계(디자인 시안)와 2단계(web 뼈대) 병행 시작. 대표 결정 필요: 감사가 지적한 것 중 대표가 정할 것 (없으면 "없음").
- [ ] **Step 3: 확인** — 보고서 '요약'+'대표 결정 필요'가 10줄 이내인지 줄 수 확인
- [ ] **Step 4: Commit + push** — 다음 날 아침 알림이 이 보고서를 읽는다
