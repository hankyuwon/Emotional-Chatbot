Final Model 1
- 
```
# Custom Data / emo_token + SENT + Q_TKN + q + SENT + A_TKN + a

user > 입사 새내기지만 우리 팀을 잘 이끌어 주시는 팀장님이 있어서 감사함을 느껴.
>>> Predicted Emotion is "감사하는"
Chatbot > 팀장님께 고마움을 표현할 방법이 있을까요


user > 그 분이 보기와는 달리 실제로는 인간미 넘치고 주어진 업무는 엄청 깔끔하게 하셔.
>>> Predicted Emotion is "감사하는"
Chatbot > 이미지와는 다른 모습이 있군요


user > 내가 더 믿고 신뢰해야지. 나는 그분께 정말 인정받는 신입이 되고 싶어.
>>> Predicted Emotion is "신뢰하는"
Chatbot > 신뢰하는 제일 큰 이유는 무엇인가요

```

Final Model 2
-
```
# Original Data / Q_TKN + q + SENT + A_TKN + a

user > 입사 새내기지만 우리 팀을 잘 이끌어 주시는 팀장님이 있어서 감사함을 느껴.
Chatbot > 팀장님께 고마우시군요

user > 그 분이 보기와는 달리 실제로는 인간미 넘치고 주어진 업무는 엄청 깔끔하게 하셔.
Chatbot > 자신의 업무가 너무 깔끔하게 처리되어 기분이 좋으시군요

user > 내가 더 믿고 신뢰해야지. 나는 그분께 정말 인정받는 신입이 되고 싶어.
Chatbot > 신뢰받는 신입이 되고 싶으시군요
```