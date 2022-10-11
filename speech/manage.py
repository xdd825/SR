# -*-coding:utf-8-*-
from flask import Flask, render_template
from flask import request
from flask_cors import CORS
import time
import random
from Speaker_Recognition import register, speakerrecog 
from Speaker_Recognition import Clasification


app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins='*')
CORS(app)

# 首頁
@app.route("/")
def index():
	return render_template("index.html")

# 聲紋註冊
@app.route("/speech",methods=['GET', 'POST'])
def beginRecorder():
	# printName = request.form["printNames"]
	printName = request.form.get('printNames')
	begin  = time.time()
	register.train_model(printName)
	# speechRecorder.run()
	return "200"

# 結束錄音
@app.route("/stopSpeech", methods=["GET", "POST"])
def stopRecorder():
	print("停止錄音......")
	# speechRecorder.stop()
	end = time.time()
	return "200"

# 聲紋辨識
@app.route("/recognize", methods=['GET', 'POST'])
def recognize():
	return speakerrecog.speakerRecog()

@app.route("/clif", methods=['GET', 'POST'])
def clif():
	return Clasification.clasRecog()

# 註冊生成文字
@app.route("/gtest", methods=['GET', 'POST'])
def gtest():
	x = random.randint(1,11)
	if x == 1:
		t = '科技越來越進步，也開始漸漸融入我們的日常生活當中，人工智慧語音技術也被廣泛應用於各領域，像是虛擬助理，能隨時接受使用者提出的需求，提供多樣化的服務' 
	if x == 2:
		t = '撥打語音客服專線，聽到客製化語音為您即時引導及諮詢。然而這些聲音都不是真實的，是研究人員透過語音合成技術達到的，運用更龐大的語音資料量進行訓練。'
	if x == 3:
		t = '並且經過多年的發展，訓練大量資料集、改良聲學特徵預測器和編碼器等等，使機器合成的聲音不僅能夠達成一般人說話的水平，更能在不同環境下，賦予聲音個性和豐富情感。'
	if x == 4:
		t = '此技術越來越純熟，雖然深度偽造技術備受爭議，但技術本身是中立的，因此所要做的，不是禁止此技術的應用，而是該審慎思考如何有效避免該技術的濫用。'
	if x == 5:
		t = '人工智慧延伸的技術，屬於如何預測未來，智慧領域中商業產出最大價值的技術，著重於從給定的資料中學習並訓練，並且根據訓練時的經驗加以改進，不是傳統的根據明確的指令運行程式碼。'
	if x == 6:
		t = '目前許多衡量語音合成模型真實還原度，皆是透過是平均意見評分，簡單來說，即為一個人對聲音品質質量的評價，共有五個分數，一是最差五是最好，加總平均後得出的分數。'
	if x == 7:
		t = '如果是要吵架,彼此只顧著反擊對方就好了,如果是要解決問題,就應該誠心去理解對方的想法,重複對方的話,一方面可以讓對方放心,另一方面,也可以把對方的意思消化一下。'
	if x == 8:
		t = '人間充滿許多的因緣,每一個因緣都可能將自己推向另一個高峰,不要輕忽任何一個人,也不要疏忽任何一個可以助人的機會,學習對每一個人都熱情以待、把每一件事都做到完善、對每一個機會都充滿感激。'
	if x == 9:
		t = '七/三八/五五定律,在整體表現上,旁人對你的觀感,只有7%取決於你真正談話的內容;而有38%在於輔助表達這些話的方法;高達55%的比重決定於,你看起來夠不夠具有說服力。'
	if x == 10:
		t = '抱歉我是個渣男,是個毒蟲、是個垃圾,但是妳會唱我的歌,夜深人靜聽著,被洗腦的情歌,如果可以,我想和妳回到那天相遇,讓時間停止那一場雨,只想擁抱妳在身邊的證據,吻妳的呼吸,一眨眼一瞬間,妳說好就是永遠不會變'
	if x == 11:
		t = '泰山的叢林法則,可以運用到現實的社會中,我們可以發現,真正成功的人,絕不只靠自身的實力,其實他更懂得整合人際資源,進而創造更多價值. 能夠了解可運用的資源,去發展良好的關係。'	
	return t

# 辨識生成文字
@app.route("/gtest1", methods=['GET', 'POST'])
def gtest1():
	x1 = random.randint(1,11)
	if x1 == 1:
		t1 = '安排勞動條件更好的工作，能夠賺取更多的金錢，才會發現行為逃逸問題。'
	if x1 == 2:
		t1 = '移工因產業不同，對薪資滿意度不同，產生的逃逸行為之評價及影響情形。'
	if x1 == 3:
		t1 = '外在因素對移工產生逃逸行為，抱持的觀感是負面或是正面的態度。'
	if x1 == 4:
		t1 = '抱持支持的觀感越強烈，就會導致主觀規範越來越強烈，增加此意圖。'
	if x1 == 5:
		t1 = '勞工經醫師證明醫療期間屆，皆享有繼續保障勞工普通事故保險。'
	if x1 == 6:
		t1 = '不同的專業技能，有不同的勞動權益及薪資對待，白領提供腦力勞動。'
	if x1 == 7:
		t1 = '移工定義為合法管道進入我國境內受雇主聘用，而從事工作以獲得工資者。'
	if x1 == 8:
		t1 = '專線即時提供發生職業災害的相關協助，補助地方政府設置服務中心。'	
	if x1 == 9:
		t1 = '白領提供腦力勞動，從事解決問題的勞動者，藍領為生產工作者。'	
	if x1 == 10:
		t1 = '因素影響行為人產生行為意圖，進而影響後續行為人是否會執行此行為。'	
	if x1 == 11:
		t1 = '感覺剝奪造成的資訊不足，也會讓人產生無法忍受的不安和痛苦。'				
	return t1

if __name__ == '__main__':
	app.run(debug = True, host='0.0.0.0', port=3000, threaded = True)
