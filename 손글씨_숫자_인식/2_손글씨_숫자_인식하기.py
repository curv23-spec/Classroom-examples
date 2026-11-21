# -*- coding: utf-8 -*-
"""
==========================================================
 2단계: 마우스로 숫자를 쓰면 인식하기 (Inference / 추론)
==========================================================

이 파일이 하는 일
-----------------
1. 1단계에서 저장한 모델(Model) 파일을 불러옵니다.
2. 마우스로 숫자를 쓸 수 있는 그림판 창(GUI, Graphical User Interface)을 엽니다.
3. 쓴 그림을 MNIST 사진과 똑같은 모양(28 × 28 픽셀)으로 바꿉니다. (전처리, Preprocessing)
4. 모델에게 "이 숫자가 뭐야?" 하고 물어보고, 답과 확률(Probability)을 보여 줍니다.

실행 방법
---------
    python 2_손글씨_숫자_인식하기.py

사용 방법
---------
  - 검은 칸에 마우스 왼쪽 버튼을 누른 채로 숫자 하나(0~9)를 크게 씁니다.
  - 마우스 버튼을 떼면 바로 인식 결과가 오른쪽에 나옵니다.
  - [지우기] 버튼(또는 키보드 Delete / 마우스 오른쪽 클릭)으로 다시 쓸 수 있습니다.

※ 반드시 '1_모델_학습하기.py'를 먼저 실행해서 모델 파일을 만들어 두어야 합니다.
"""

# ----------------------------------------------------------
# 필요한 도구(라이브러리, Library) 불러오기
# ----------------------------------------------------------
import sys
from pathlib import Path
import tkinter as tk                          # 파이썬에 기본으로 들어 있는 창(GUI) 만들기 도구
from tkinter import messagebox                # 알림 창을 띄우는 도구

import numpy as np                            # 숫자 배열(Array) 계산 도구
import joblib                                 # 저장한 모델 파일을 불러오는 도구
from PIL import Image, ImageDraw              # 그림(Image)을 만들고 다루는 도구 (Pillow)

# ----------------------------------------------------------
# 설정값(상수, Constant)
# ----------------------------------------------------------
현재_폴더 = Path(__file__).resolve().parent
모델_파일_경로 = 현재_폴더 / "숫자_인식_모델.joblib"

그림판_크기 = 280        # 그림판은 280 × 280 픽셀 (MNIST 28 × 28의 10배)
펜_굵기 = 22             # 붓(펜) 굵기. 너무 가늘면 28칸으로 줄였을 때 글씨가 흐려집니다.


# ==========================================================
# 전처리(Preprocessing): 내가 쓴 그림 → MNIST 모양 사진
# ==========================================================
def 그림을_MNIST_모양으로_바꾸기(그림):
    """
    모델은 MNIST 사진만 보고 배웠기 때문에, 우리가 쓴 그림도
    MNIST와 '같은 규칙'으로 바꿔 줘야 잘 알아봅니다.

    MNIST 사진의 규칙
      ① 검은 배경에 흰 글씨
      ② 숫자는 가로·세로 중 긴 쪽이 20픽셀이 되도록 크기를 맞춤 (가로세로 비율은 유지)
      ③ 그 숫자를 28 × 28 칸 안에 넣되, 무게중심(Center of mass)이 가운데 오도록 배치
      ④ 밝기 값은 0~1 사이로 정규화(Normalization)

    매개변수(Parameter)
      그림 : 280 × 280 흑백(Grayscale) PIL 이미지

    반환값(Return value)
      (모델 입력용 1 × 784 배열, 사람이 볼 수 있는 28 × 28 배열)
      아무것도 안 썼으면 (None, None)
    """
    배열 = np.array(그림, dtype=np.float32)   # 그림을 숫자 표(배열)로 바꿉니다. 0=검정, 255=흰색

    # --- ② 숫자가 있는 부분만 잘라 내기 (경계 상자, Bounding box) ---
    글씨가_있는_행 = np.where(배열.max(axis=1) > 0)[0]   # 흰 점이 하나라도 있는 가로줄들
    글씨가_있는_열 = np.where(배열.max(axis=0) > 0)[0]   # 흰 점이 하나라도 있는 세로줄들
    if len(글씨가_있는_행) == 0:
        return None, None                                 # 아무것도 안 썼음

    위, 아래 = 글씨가_있는_행[0], 글씨가_있는_행[-1]
    왼쪽, 오른쪽 = 글씨가_있는_열[0], 글씨가_있는_열[-1]
    잘라낸_그림 = 그림.crop((왼쪽, 위, 오른쪽 + 1, 아래 + 1))

    # 긴 쪽이 20픽셀이 되도록 줄이기 (가로세로 비율, Aspect ratio 유지)
    가로, 세로 = 잘라낸_그림.size
    비율 = 20.0 / max(가로, 세로)
    새_가로 = max(1, round(가로 * 비율))
    새_세로 = max(1, round(세로 * 비율))
    # LANCZOS: 사진을 줄일 때 부드럽게(안티에일리어싱, Anti-aliasing) 줄여 주는 방법
    작은_그림 = 잘라낸_그림.resize((새_가로, 새_세로), Image.LANCZOS)

    # --- ③ 28 × 28 검은 도화지 한가운데에 붙이기 ---
    도화지 = Image.new("L", (28, 28), color=0)
    도화지.paste(작은_그림, ((28 - 새_가로) // 2, (28 - 새_세로) // 2))
    작은_배열 = np.array(도화지, dtype=np.float32)

    # 무게중심(Center of mass)을 구해서 정확히 가운데(14, 14)로 옮기기
    #   - 밝은 픽셀일수록 '무게'가 크다고 생각하고 평균 위치를 구합니다.
    #   - 예: 숫자 7은 위쪽이 무거워서, 상자 가운데와 무게중심이 다릅니다.
    전체_무게 = 작은_배열.sum()
    행_번호, 열_번호 = np.indices(작은_배열.shape)
    중심_행 = (행_번호 * 작은_배열).sum() / 전체_무게
    중심_열 = (열_번호 * 작은_배열).sum() / 전체_무게
    이동_행 = int(round(14 - 중심_행))
    이동_열 = int(round(14 - 중심_열))
    # np.roll: 배열을 통째로 밀어서 옮깁니다. (가장자리는 비어 있으니 안전합니다)
    작은_배열 = np.roll(작은_배열, shift=(이동_행, 이동_열), axis=(0, 1))

    # --- ④ 정규화: 0~255 → 0~1 (1단계 학습 때와 똑같이!) ---
    작은_배열 = 작은_배열 / 255.0

    # 모델은 사진 한 장을 '한 줄로 펼친 784개 숫자'로 받습니다. (평탄화, Flatten)
    모델_입력 = 작은_배열.reshape(1, 784)
    return 모델_입력, 작은_배열


# ==========================================================
# 그림판 프로그램 (GUI)
# ==========================================================
class 숫자_인식_그림판:
    """마우스로 숫자를 쓰고, 그 결과를 보여 주는 창 하나를 만드는 설계도(클래스, Class)"""

    def __init__(self, 창, 모델):
        self.창 = 창
        self.모델 = 모델
        self.이전_위치 = None   # 선을 이어 그리기 위해 바로 전 마우스 위치를 기억합니다.

        창.title("손글씨 숫자 인식기 (Handwritten Digit Recognizer)")
        창.configure(bg="#f4f4f4")
        창.resizable(False, False)

        # 화면에 보이는 그림판과 별개로, 모델에게 줄 '진짜 그림'을 메모리에 같이 그립니다.
        # "L" 모드 = 흑백(Grayscale), 0 = 검은색으로 채움
        self.그림 = Image.new("L", (그림판_크기, 그림판_크기), color=0)
        self.붓 = ImageDraw.Draw(self.그림)

        self._화면_만들기()

    # ------------------------------------------------------
    # 화면 구성 (Layout)
    # ------------------------------------------------------
    def _화면_만들기(self):
        글꼴 = "Malgun Gothic"   # 윈도우 기본 한글 글꼴 '맑은 고딕'

        # [왼쪽] 숫자를 쓰는 곳
        왼쪽 = tk.Frame(self.창, bg="#f4f4f4")
        왼쪽.grid(row=0, column=0, padx=16, pady=16, sticky="n")

        tk.Label(왼쪽, text="여기에 숫자 하나를 크게 써 보세요 ✍",
                 font=(글꼴, 12, "bold"), bg="#f4f4f4").pack(anchor="w")

        # 캔버스(Canvas): 그림을 그릴 수 있는 칸
        self.캔버스 = tk.Canvas(왼쪽, width=그림판_크기, height=그림판_크기,
                               bg="black", highlightthickness=2,
                               highlightbackground="#888", cursor="pencil")
        self.캔버스.pack(pady=(6, 8))

        # 마우스 이벤트(Event) 연결: 어떤 동작을 하면 어떤 함수를 부를지 정합니다.
        self.캔버스.bind("<ButtonPress-1>", self.그리기_시작)     # 왼쪽 버튼 누름
        self.캔버스.bind("<B1-Motion>", self.그리는_중)           # 누른 채로 움직임
        self.캔버스.bind("<ButtonRelease-1>", self.그리기_끝)     # 버튼 뗌 → 인식!
        self.캔버스.bind("<Button-3>", lambda 이벤트: self.지우기())  # 오른쪽 클릭 → 지우기
        self.창.bind("<Delete>", lambda 이벤트: self.지우기())         # Delete 키 → 지우기
        self.창.bind("<Escape>", lambda 이벤트: self.창.destroy())     # Esc 키 → 종료

        버튼줄 = tk.Frame(왼쪽, bg="#f4f4f4")
        버튼줄.pack(fill="x")
        tk.Button(버튼줄, text="지우기 (Clear)", font=(글꼴, 11), width=14,
                  command=self.지우기).pack(side="left")
        tk.Button(버튼줄, text="인식하기 (Predict)", font=(글꼴, 11), width=14,
                  command=self.인식하기).pack(side="right")

        # [오른쪽] 결과를 보여 주는 곳
        오른쪽 = tk.Frame(self.창, bg="#f4f4f4")
        오른쪽.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="n")

        tk.Label(오른쪽, text="인식 결과 (Prediction)",
                 font=(글꼴, 12, "bold"), bg="#f4f4f4").pack(anchor="w")
        self.결과_글자 = tk.Label(오른쪽, text="?", font=("Arial", 64, "bold"),
                                fg="#1f6feb", bg="#f4f4f4", width=2)
        self.결과_글자.pack()
        self.확신도_글자 = tk.Label(오른쪽, text="숫자를 써 주세요",
                                 font=(글꼴, 11), bg="#f4f4f4")
        self.확신도_글자.pack(pady=(0, 8))

        # 0~9 각각의 확률을 막대그래프(Bar chart)로 보여 줄 캔버스
        tk.Label(오른쪽, text="숫자별 확률 (Probability)",
                 font=(글꼴, 10), bg="#f4f4f4").pack(anchor="w")
        self.막대_캔버스 = tk.Canvas(오른쪽, width=260, height=200,
                                   bg="white", highlightthickness=0)
        self.막대_캔버스.pack(pady=(2, 8))

        # 모델이 실제로 본 28 × 28 사진 (전처리 결과)
        tk.Label(오른쪽, text="모델이 본 28×28 사진",
                 font=(글꼴, 10), bg="#f4f4f4").pack(anchor="w")
        self.미리보기_캔버스 = tk.Canvas(오른쪽, width=84, height=84, bg="black",
                                      highlightthickness=1, highlightbackground="#888")
        self.미리보기_캔버스.pack(anchor="w", pady=2)

        self._막대그래프_그리기(np.zeros(10))

    # ------------------------------------------------------
    # 마우스로 그리기
    # ------------------------------------------------------
    def 그리기_시작(self, 이벤트):
        self.이전_위치 = (이벤트.x, 이벤트.y)
        self._점_찍기(이벤트.x, 이벤트.y)

    def 그리는_중(self, 이벤트):
        x, y = 이벤트.x, 이벤트.y
        if self.이전_위치 is not None:
            x0, y0 = self.이전_위치
            # 화면에 선 긋기 (보이는 그림)
            self.캔버스.create_line(x0, y0, x, y, fill="white", width=펜_굵기,
                                   capstyle=tk.ROUND, smooth=True)
            # 메모리 그림에도 똑같이 선 긋기 (모델이 볼 그림)
            self.붓.line([x0, y0, x, y], fill=255, width=펜_굵기)
        self._점_찍기(x, y)   # 선 끝을 둥글게 만들어 이음새가 끊기지 않게 합니다.
        self.이전_위치 = (x, y)

    def 그리기_끝(self, 이벤트):
        self.이전_위치 = None
        self.인식하기()   # 손을 떼면 자동으로 인식합니다.

    def _점_찍기(self, x, y):
        r = 펜_굵기 / 2
        self.캔버스.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.붓.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def 지우기(self):
        self.캔버스.delete("all")
        self.붓.rectangle([0, 0, 그림판_크기, 그림판_크기], fill=0)
        self.결과_글자.config(text="?")
        self.확신도_글자.config(text="숫자를 써 주세요")
        self.미리보기_캔버스.delete("all")
        self._막대그래프_그리기(np.zeros(10))

    # ------------------------------------------------------
    # 인식하기 (추론, Inference)
    # ------------------------------------------------------
    def 인식하기(self):
        모델_입력, 작은_배열 = 그림을_MNIST_모양으로_바꾸기(self.그림)
        if 모델_입력 is None:
            return   # 아무것도 안 썼으면 그냥 넘어갑니다.

        # predict_proba(): 0~9 각각일 확률을 돌려줍니다. 열 개를 다 더하면 1(=100%)
        확률들 = self.모델.predict_proba(모델_입력)[0]
        정답_후보 = int(np.argmax(확률들))   # argmax: 가장 큰 값이 있는 자리 = 가장 그럴듯한 숫자
        확신도 = 확률들[정답_후보]

        self.결과_글자.config(text=str(정답_후보))
        if 확신도 >= 0.8:
            설명 = "확실해요!"
        elif 확신도 >= 0.5:
            설명 = "아마도요."
        else:
            설명 = "잘 모르겠어요. 다시 써 볼까요?"
        self.확신도_글자.config(text=f"확신도(Confidence) {확신도 * 100:.1f}% · {설명}")

        self._막대그래프_그리기(확률들)
        self._미리보기_그리기(작은_배열)

    def _막대그래프_그리기(self, 확률들):
        c = self.막대_캔버스
        c.delete("all")
        막대_폭, 간격, 바닥, 최대_높이 = 18, 8, 180, 160
        가장_큰_자리 = int(np.argmax(확률들)) if 확률들.max() > 0 else -1
        for 숫자, 확률 in enumerate(확률들):
            x0 = 10 + 숫자 * (막대_폭 + 간격)
            높이 = 확률 * 최대_높이
            색 = "#1f6feb" if 숫자 == 가장_큰_자리 else "#b6c6e3"
            c.create_rectangle(x0, 바닥 - 높이, x0 + 막대_폭, 바닥, fill=색, outline="")
            c.create_text(x0 + 막대_폭 / 2, 바닥 + 10, text=str(숫자), font=("Arial", 9))
            if 확률 >= 0.05:
                c.create_text(x0 + 막대_폭 / 2, 바닥 - 높이 - 8,
                              text=f"{확률 * 100:.0f}", font=("Arial", 7), fill="#555")

    def _미리보기_그리기(self, 작은_배열):
        c = self.미리보기_캔버스
        c.delete("all")
        칸 = 3   # 28칸 × 3배 = 84픽셀
        for 행 in range(28):
            for 열 in range(28):
                밝기 = int(작은_배열[행, 열] * 255)
                if 밝기 > 0:
                    색 = f"#{밝기:02x}{밝기:02x}{밝기:02x}"
                    c.create_rectangle(열 * 칸, 행 * 칸, (열 + 1) * 칸, (행 + 1) * 칸,
                                       fill=색, outline="")


# ==========================================================
# 프로그램 시작점
# ==========================================================
def main():
    # 모델 파일이 없으면 1단계를 먼저 하라고 알려 줍니다.
    if not 모델_파일_경로.exists():
        tk.Tk().withdraw()
        messagebox.showerror("모델 파일 없음",
                             "모델 파일이 없습니다.\n먼저 '1_모델_학습하기.py'를 실행해 주세요.")
        sys.exit(1)

    print("📂 모델을 불러오는 중...")
    모델 = joblib.load(모델_파일_경로)
    print("✅ 준비 완료! 창에 숫자를 써 보세요. (종료: 창 닫기 또는 Esc)")

    창 = tk.Tk()
    숫자_인식_그림판(창, 모델)
    창.mainloop()   # 창을 띄우고, 사용자가 닫을 때까지 계속 기다립니다. (이벤트 루프, Event loop)


if __name__ == "__main__":
    main()
