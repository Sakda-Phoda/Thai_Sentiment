# Error Analysis — WangchanBERTa (test set)

- Overall error rate: **31.8%** (849/2671)
- Errors with English text (code-mixing): **164**
- Errors with emoji: **61**
- Most confused pair (true → pred): **neu → pos (311 times)**
- Minority class `q` misclassified: **19**


## Top confusion pairs (true → pred)

| True | Pred | Count |
|------|------|-------|
| neu | pos | 311 |
| neu | neg | 169 |
| pos | neu | 133 |
| neu | q | 75 |
| neg | neu | 70 |
| neg | pos | 33 |
| pos | neg | 31 |
| q | neu | 13 |

## Top 'Confident but Wrong' examples (Top 20 by confidence)

| Text | True | Pred | Conf |
|------|------|------|------|
| โปรเฉพาะแค่6ชิ้นหรอ ไปซื้อมาจะซื้อ10ชิ้น พนง.บอกไม่ร่วมรายการ แถมหน้าเหวี่ยงไปอี้ก 😏 | neu | neg | 0.99 |
| รีวิวรองพื้น Marc Jacobซะหน่อย เมื่อวานไปสอยตัวใหม่มา ขวดแบนๆ Shameless foundation คือตัวนี้ตอนลองแบบ เห้ย ทาแล้ว????? ม… | pos | neg | 0.99 |
| หาซื้อผ้าอนามัยลอรีเอะ ซูเปอร์ เจนเทิล พลัส ได้ที่ 7-Eleven ทุกสาขาได้เลยนะคะ :) http://www.lauriermybrand.com/2013/l-pr… | pos | neu | 0.99 |
| เดี๋ยวนี้พนักงานที่สนามบินบางคนทำไมถึงพูดจาและแสดงกิริยาออกมาแย่จัง เห็นเป็นต่างชาติแล้วเหวี่ยงภาษาไทย ถึงเค้าจะฟังไม่ออ… | pos | neg | 0.99 |
| ป่านนี้กุยังไม่เจอ 4u2 ไม่อยากสั่งออนไลน์ | neu | neg | 0.99 |
| ธนาธร - ยอดอัลบั้มสูง - ดิจิตอลกาก แม้ว - ยอดอัลบั้มไม่รู้ - แต่ดิจิตอลพุ่งกระชูด ลุง - ค่ายดันจนน่ารำคาญไอซั๊ซ อภิสิทธิ… | neu | neg | 0.99 |
| วันนี้ไปทาน สาขาบิ๊กซีหัวหมาก พนักงานบริการดีค่ะ แต่เสียแค่เรื่องการเติมอาหาร ช้ามากกกกค่ะ | pos | neg | 0.99 |
| ไม่รู้ว่ะรู้แต่กินอะไรไม่อร่อย..แดร่กสปาย4ขวดเมานอนนับดาว | neu | neg | 0.99 |
| บาร์บีก้อนเละอ่ะ บอกเลย 😑 | neu | neg | 0.99 |
| จิ๋ว เชียงราย - ต๋อง มัตซูชิ รอบ 16 คน ระบบ 5/9 เฟรม 'SangSom 6 Red Open Grand Final 2017' @ Terminal 21 Korat | pos | neu | 0.99 |
| ตอบ สินค้านี้ ลอรีเอะ ซุปเปอร์ เจนเทิลพลัส 25-30 ซม. 6-7ชิ้น ราคา 21 บาท | pos | neu | 0.99 |
| etude ค่า | neu | q | 0.99 |
| สนามบินที่มิลานโนอิตาลี ราคาไม่เป็นแพงราคาก็เหมือนข้างนอกเพราเรากินแมคโดนัท ดูราคาแล้วเท่ากัน | neu | neg | 0.99 |
| นกแอร์กับแอร์เอเชียเที่ยวบินไประนองยกเลิกมั้ยค่ะ | neu | q | 0.98 |
| น้ำหอม dior เท่าไหร่ค่ะ | neu | q | 0.98 |
| /Cetaphil Oily Skin Cleanser/Acne-Aid liquid cleanser ตอนนี้มีสามตัวนี้ละ ต้องซื้อยี่ห้อไหนอีกมะ 555 | neu | q | 0.98 |
| มันบ่แซบ | neu | neg | 0.98 |
| มีปล่อยอิสระบ้าง แต่ส่วนมากไม่ถึงชั่วโมง เป็นเวลากินข้าว หรือไม่ก็ให้นั่งให้ยืนโง่ๆ การบรีฟงานก็ห่วย ไม่ต้องบรีฟก็ได้ ถึ… | neu | neg | 0.98 |
| มี biotherm emulsion มั้ยค่ะ | neu | q | 0.98 |
| บ่แซบ | neu | neg | 0.98 |

## Summary of Patterns Found
1. **Most confused pair is neu → pos (311 times)** — Boundary between these 2 classes is thin
2. **Minority class `q` is misclassified**: Insufficient training data, model hasn't seen enough examples
3. **code-mixing / emoji / sarcasm**: Context where Thai subwords fail to fully capture sentiment

## Future Improvements
- **Data augmentation** for minority class `q` (back-translation / templating) to solve data scarcity
- Group emojis as specific tokens / add emoji sentiment features
- Try larger models (e.g., larger wangchanberta) or ensemble with baseline
- Collect more sarcasm data or perform targeted hard-example mining
