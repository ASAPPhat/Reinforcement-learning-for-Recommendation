# Đặt tên cột theo mô tả của dataset
RATINGS FILE DESCRIPTION
================================================================================

All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

USERS FILE DESCRIPTION
================================================================================

User information is in the file "users.dat" and is in the following
format:

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is
not checked for accuracy.  Only users who have provided some demographic
information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

MOVIES FILE DESCRIPTION
================================================================================

Movie information is in the file "movies.dat" and is in the following
format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western

- Some MovieIDs do not correspond to a movie due to accidental duplicate
entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist

# Mã hóa ID và định nghĩa MASK/PAD

Số 0: Dành riêng làm khoảng trống ([PAD]) để bù vào các chuỗi phim bị ngắn.

Số 1 đến 3706: Dành cho các bộ phim có thật (đáp án).

Số 3707 (len + 1): Dành riêng làm Token che giấu ([MASK]), báo hiệu cho AI biết đây là "chỗ trống cần suy luận".


Tạo số 3707(blank) để nó che các phim trong chuỗi lại ví dụ

Giả sử ta chọn số 2 (phim Jumanji) làm ký hiệu để che.

-Chuỗi gốc người dùng xem: [Toy Story (1), Heat (6), Casino (16)]

-Hành động: Bạn muốn kiểm tra mô hình bằng cách giấu phim Heat (6) đi.

-Nếu bạn dán số 2 đè lên số 6: Mảng dữ liệu biến thành [1, 2, 16].

->Hậu quả: Khi máy tính đọc mảng [1, 2, 16], nó sẽ hiểu là: "À, người dùng này đã xem Toy Story, sau đó xem Jumanji, rồi xem Casino. Một chuỗi rất bình thường, chẳng có câu hỏi nào cần giải quyết ở đây cả!". Mô hình hoàn toàn không biết rằng vị trí ở giữa đã bị thay đổi và cần phải được dự đoán. Nó đã bị "đánh lừa" thành một chuỗi hợp lệ khác.

Giải pháp:

Thay vì dùng số 2, ta dùng một con số nằm ngoài danh sách phim có thật là 3707.

Chuỗi gốc: [Toy Story (1), Heat (6), Casino (16)]

Hành động: Bạn dán miếng băng keo 3707 đè lên phim Heat (6).

Kết quả: Mảng dữ liệu biến thành [1, 3707, 16].

Máy tính sẽ hiểu: "Khoan đã! Từ điển phim chỉ có từ 1 đến 3706 thôi. Số 3707 này là một ký hiệu đặc biệt báo hiệu chỗ trống (Fill-in-the-blank). Nhiệm vụ của mình là phải nhìn vào phim số 1 và phim số 16 ở hai bên, để suy luận xem cái phim nằm dưới lớp băng keo 3707 kia thực chất là phim số mấy trong tập từ 1 đến 3706!"

# Gom nhóm thành chuỗi lịch sử (Sequence Generation)

Gom các đánh giá rời rạc thành các chuỗi hành vi (sequences) của từng người dùng

**Output mẫu**

{

    1: [32, 15, 114, 5, ...], # Lịch sử xem của UserID 1
    
    2: [18, 5, 233, ...],     # Lịch sử xem của UserID 2
    ...
}

# Chia dữ liệu train test val

-Tập Train lấy N-2 phim đầu, phần còn lại cho Val và Test. Nếu chia ngẫu nhiên, AI có thể nhìn thấy phim người dùng xem ở tháng 12 (tương lai) để đi lùi lại dự đoán phim họ xem ở tháng 10 (quá khứ).

Do đó, cách chia bắt buộc phải đi dọc theo dòng thời gian:Tập Test (Bộ phim cuối cùng - $N$): Đây là bài thi thật. "Với tất cả những gì người dùng đã xem từ trước tới nay, hãy đoán xem NGAY BÂY GIỜ họ muốn xem phim gì?".Tập Val (Bộ phim áp chót - $N-1$): Đây là bài thi thử. Trong lúc huấn luyện, AI lấy phim áp chót ra để tự kiểm tra xem mình học tốt chưa (phục vụ cho việc Early Stopping - dừng sớm để tránh học vẹt).Tập Train (Toàn bộ phần còn lại - từ $1$ đến $N-2$): Đây là sách giáo khoa để AI học cách tìm ra quy luật sở thích của người dùng.

# Kỹ thuật Che ngẫu nhiên (Random Masking trên tập Train)
Mục đích: Tạo ra các bài tập "điền vào chỗ trống" (Cloze Task) để AI rèn luyện.

Cách hoạt động: Đi dọc theo tập Train (sách giáo khoa), ta dùng thuật toán tung đồng xu để che ngẫu nhiên khoảng 15% số phim bằng MASK_TOKEN. Nhờ đó, AI bị ép phải nhìn cả mảng phim bên trái lẫn bên phải để suy luận ra phim bị che, phát huy tối đa sức mạnh của mạng Transformer hai chiều.

# Đóng gói kích thước cố định (Padding & Truncating)
Mạng Nơ-ron (Neural Network) không thể tính toán nếu lúc thì nhận vào một chuỗi 10 phim, lúc thì nhận chuỗi 200 phim. Ma trận toán học yêu cầu một kích thước cố định (ví dụ: max_len = 50).

Truncating (Cắt bớt): Nếu chuỗi dài hơn 50, ta chỉ giữ lại 50 phim gần nhất (phần đuôi). Lý do: Sở thích xem phim hiện tại của con người quan trọng và phản ánh đúng nhất so với sở thích từ chục năm trước.

Padding (Bù trống): Nếu chuỗi ngắn hơn 50, ta chèn thêm các số 0 (Token [PAD]) vào đầu chuỗi để lấp đầy ma trận, giúp GPU xử lý tính toán đồng loạt (Batch processing) mà không bị lỗi.
