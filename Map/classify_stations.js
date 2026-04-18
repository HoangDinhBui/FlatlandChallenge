const data = require('./stations_raw.json');

const thongNhat = [
  'Ga Hà Nội','Ga Giáp Bát','Ga Văn Điển','Ga Thường Tín','Ga Chợ Tía','Ga Vạn Điểm',
  'Ga Phú Xuyên','Ga Đồng Văn','Ga Phủ Lý','Ga Thịnh Châu','Ga Bình Lục','Ga Cầu Họ','Ga Đặng Xá',
  'Ga Nam Định','Ga Trình Xuyên','Ga Núi Gôi','Ga Cát Đằng','Ga Ninh Bình',
  'Ga Cầu Yên','Ga Ghềnh','Ga Đồng Giao','Ga Bỉm Sơn','Ga Đò Lèn','Ga Nghĩa Trang','Ga Thanh Hóa',
  'Ga Yên Thái','Ga Minh Khôi','Ga Thị Long','Ga Văn Trai','Ga Khoa Trường','Ga Trường Lâm',
  'Ga Hoàng Mai','Ga Cầu Giát','Ga Yên Lý','Ga Chợ Sy','Ga Mỹ Lý','Ga Nghi Long','Ga Quán Hành','Ga Vinh',
  'Ga Yên Xuân','Ga Yên Trung','Ga Đức Lạc','Ga Yên Duệ','Ga Hòa Duyệt','Ga Thanh Luyện',
  'Ga Chu Lễ','Ga Hương Phố','Ga Phúc Trạch','Ga La Khê','Ga Tân Ấp','Ga Đồng Chuối',
  'Ga Kim Lũ','Ga Đồng Lê','Ga Ngọc Lâm','Ga Lạc Sơn','Ga Lệ Sơn','Ga Minh Lệ',
  'Ga Ngân Sơn','Ga Thọ Lộc','Ga Hoàn Lão','Ga Phúc Tự','Ga Đồng Hới',
  'Ga Lệ Kỳ','Ga Long Đại','Ga Mỹ Đức','Ga Phú Hòa','Ga Mỹ Trạch','Ga Thượng Lâm',
  'Ga Sa Lung','Ga Tiên An','Ga Hà Thanh','Ga Đông Hà','Ga Quảng Trị','Ga Diên Sanh',
  'Ga Mỹ Chánh','Ga Phò Trạch','Ga Hiền Sỹ','Ga Văn Xá','Ga Huế','Ga An Cựu',
  'Ga Hương Thuỷ','Ga Truồi','Ga Cầu Hai','Ga Thừa Lưu','Ga Lăng Cô',
  'Ga Hải Vân Bắc','Ga Hải Vân','Ga Hải Vân Nam','Ga Kim Liên','Ga Thanh Khê','Ga Đà Nẵng',
  'Ga Lệ Trạch','Ga Nông Sơn','Ga Trà Kiệu','Ga Phú Cang','Ga Tam Thành','Ga An Mỹ','Ga Tam Kỳ',
  'Ga Diêm Phổ','Ga Núi Thành','Ga Trị Bình','Ga Bình Sơn','Ga Đại Lộc','Ga Quảng Ngãi',
  'Ga Hòa Vinh Tây','Ga Mộ Đức','Ga Thạch Trụ','Ga Đức Phổ','Ga Thủy Trạch','Ga Sa Huỳnh',
  'Ga Tam Quan','Ga Bồng Sơn','Ga Vạn Phú','Ga Phù Mỹ','Ga Khánh Phước','Ga Phù Cát',
  'Ga Bình Định','Ga Diêu Trì','Ga Quy Nhơn',
  'Ga Tân Vinh','Ga Vân Canh','Ga Phước Lãnh','Ga La Hai','Ga Chí Thạnh','Ga Hòa Đa',
  'Ga Tuy Hòa','Ga Đông Tác','Ga Phú Hiệp','Ga Hảo Sơn','Ga Đại Lãnh','Ga Tu Bông',
  'Ga Giã','Ga Hòa Huỳnh','Ga Ninh Hòa','Ga Phong Thạnh','Ga Lương Sơn','Ga Nha Trang',
  'Ga Phú Vinh','Ga Cây Cầy','Ga Hòa Tân','Ga Suối Cát','Ga Ngã Ba','Ga Cam Thịnh Đông','Ga Cà Rôm',
  'Ga Phước Nhơn','Ga Tháp Chàm',
  'Ga Hòa Trinh','Ga Cà Ná','Ga Vĩnh Tân','Ga Vĩnh Hảo','Ga Sông Lòng Sông',
  'Ga Phong Phú (đường sắt Bắc Nam)',
  'Ga Sông Mao','Ga Châu Hanh','Ga Sông Lũy','Ga Long Thạnh','Ga Ma Lâm','Ga Hàm Liêm',
  'Ga Phan Thiết','Ga Bình Thuận',
  'Ga Hàm Cường Tây','Ga Suối Vận','Ga Sông Phan','Ga Sông Dinh','Ga Suối Kiết','Ga Gia Huynh',
  'Ga Trản Táo','Ga Gia Ray','Ga Bảo Chánh','Ga Long Khánh','Ga Dầu Giây',
  'Ga Trung Hòa (đường sắt Bắc Nam)','Ga Trảng Bom','Ga Hố Nai','Ga Biên Hòa',
  'Ga Dĩ An','Ga Sóng Thần','Ga Bình Triệu','Ga Gò Vấp','Ga Sài Gòn'
];

const hanoiLaoCai = [
  'Ga Hà Nội','Ga Long Biên','Ga Gia Lâm','Ga Yên Viên','Ga Đông Anh','Ga Bắc Hồng',
  'Ga Phúc Yên','Ga Thạch Lỗi','Ga Hương Canh','Ga Vĩnh Yên','Ga Hướng Lại',
  'Ga Bạch Hạc','Ga Việt Trì','Ga Phủ Đức','Ga Tiên Kiên','Ga Phú Thọ','Ga Chí Chủ',
  'Ga Vũ Ẻn','Ga Ấm Thượng','Ga Đoan Thượng','Ga Văn Phú','Ga Cổ Phúc',
  'Ga Yên Bái','Ga Ngòi Hóp','Ga Mậu A','Ga Mậu Đông','Ga Trái Hút','Ga Lâm Giang',
  'Ga Lang Khay','Ga Lang Thíp','Ga Bảo Hà','Ga Thái Văn','Ga Lạng','Ga Thái Niên',
  'Ga Làng Giàng','Ga Cầu Nhô','Ga Phố Lu','Ga Lào Cai'
];

const hanoiHaiPhong = [
  'Ga Hà Nội','Ga Long Biên','Ga Gia Lâm','Ga Cầu Bây','Ga Phú Thụy',
  'Ga Lạc Đạo','Ga Tuấn Lương','Ga Cẩm Giàng','Ga Hải Dương','Ga Cao Xá',
  'Ga Tiền Trung','Ga Lai Khê','Ga Phạm Xá','Ga Phú Thai','Ga Dụ Nghĩa',
  'Ga Vật Cách','Ga Thượng Lý','Ga Hải Phòng'
];

const hanoiDongDang = [
  'Ga Hà Nội','Ga Long Biên','Ga Gia Lâm','Ga Yên Viên','Ga Cổ Loa','Ga Đông Anh',
  'Ga Từ Sơn','Ga Lim','Ga Bắc Ninh','Ga Thị Cầu',
  'Ga Bắc Giang','Ga Bảo Sơn','Ga Kép',
  'Ga Sen Hồ','Ga Cẩm Lý','Ga Chí Linh','Ga Phố Tráng',
  'Ga Bắc Lệ','Ga Phố Vị','Ga Voi Xô','Ga Chi Lăng','Ga Sông Hòa',
  'Ga Đồng Mỏ','Ga Bắc Thủy','Ga Bản Thí','Ga Yên Trạch','Ga Lạng Sơn','Ga Đồng Đăng'
];

const hanoiQuanTrieu = [
  'Ga Hà Nội','Ga Long Biên','Ga Gia Lâm','Ga Yên Viên','Ga Đông Anh',
  'Ga Trung Giã','Ga Phổ Yên','Ga Đại Từ','Ga Lưu Xá','Ga Thái Nguyên','Ga Quán Triều'
];

const kepHaLong = [
  'Ga Kép','Ga Lan Mẫu','Ga Mạo Khê','Ga Đông Triều',
  'Ga Uông Bí','Ga Yên Dưỡng','Ga Bàn Cờ','Ga Yên Cư','Ga Hạ Long','Ga Cái Lân'
];

const dalatTraiMat = ['Ga Đà Lạt','Ga Trại Mát'];

const lines = {
  'Thống Nhất (Bắc-Nam)': thongNhat,
  'Hà Nội - Lào Cai': hanoiLaoCai,
  'Hà Nội - Hải Phòng': hanoiHaiPhong,
  'Hà Nội - Đồng Đăng': hanoiDongDang,
  'Hà Nội - Quán Triều': hanoiQuanTrieu,
  'Kép - Hạ Long': kepHaLong,
  'Đà Lạt - Trại Mát': dalatTraiMat
};

const stationMap = {};
data.forEach(s => { stationMap[s.name] = s; });

const classified = {};
for (const [line, names] of Object.entries(lines)) {
  classified[line] = [];
  for (const name of names) {
    if (stationMap[name]) {
      classified[line].push({ ...stationMap[name], line });
    }
  }
}

let total = 0;
for (const [line, stations] of Object.entries(classified)) {
  console.log(line + ': ' + stations.length + ' stations');
  total += stations.length;
}
console.log('Total classified:', total);

const allClassified = new Set();
for (const names of Object.values(lines)) {
  names.forEach(n => allClassified.add(n));
}
const unclassified = data.filter(s => !allClassified.has(s.name));
console.log('Unclassified:', unclassified.length);
unclassified.forEach(s => console.log('  -', s.name, s.lat, s.lon));

// Save classified data
const fs = require('fs');
fs.writeFileSync('classified_stations.json', JSON.stringify(classified, null, 2));
console.log('\nSaved classified_stations.json');
