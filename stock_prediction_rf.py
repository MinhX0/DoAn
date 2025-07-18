import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Thiết lập matplotlib
import matplotlib
print(f"Current matplotlib backend: {matplotlib.get_backend()}")

# Thiết lập backend phù hợp
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')
        print("Warning: Using non-interactive backend. Plots may not display.")

# Thiết lập font cho matplotlib để hiển thị tiếng Việt
# plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.max_open_warning'] = 50

# Bật interactive mode
plt.ion()

class StockPredictor:
    def __init__(self, symbol, lookback_days=10):
        """
        Khởi tạo bộ dự đoán cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu (VD: 'AAPL', 'GOOGL')
            lookback_days: Số ngày lookback để tạo pattern (mặc định 10 ngày)
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fetch_data(self):
        """Lấy dữ liệu cổ phiếu từ yfinance"""
        try:
            print(f"Đang lấy dữ liệu cho {self.symbol}...")
            # Lấy dữ liệu 1 năm + thêm buffer
            self.data = yf.download(self.symbol, period="1y", interval="1d")
            
            if self.data.empty:
                raise ValueError(f"Không thể lấy dữ liệu cho mã {self.symbol}")
            
            # Tính các chỉ số cơ bản
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['Price_Change'] = self.data['Close'].diff()
            self.data['Volume_Change'] = self.data['Volume'].pct_change()
            
            # Tính moving averages
            self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            
            # Tính RSI đơn giản
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # Tính volatility
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
            
            # Loại bỏ NaN
            self.data = self.data.dropna()
            
            print(f"Đã lấy thành công {len(self.data)} ngày dữ liệu cho {self.symbol}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu: {e}")
            return False
    
    def create_features(self):
        """Tạo features dựa trên pattern của n ngày trước"""
        features = []
        targets_direction = []
        targets_price = []
        
        # Tạo features từ lookback_days
        for i in range(self.lookback_days, len(self.data) - 1):
            # Features: pattern của lookback_days ngày trước
            feature_row = []
            
            # Lấy các giá trị trong lookback_days ngày
            for j in range(self.lookback_days):
                idx = i - self.lookback_days + j
                
                # Kiểm tra bounds
                if idx < 0 or idx >= len(self.data):
                    continue
                
                # Thêm các features quan trọng với kiểm tra NaN
                try:
                    returns = float(self.data['Returns'].iloc[idx]) if pd.notna(self.data['Returns'].iloc[idx]) else 0.0
                    volume_change = float(self.data['Volume_Change'].iloc[idx]) if pd.notna(self.data['Volume_Change'].iloc[idx]) and not np.isinf(self.data['Volume_Change'].iloc[idx]) else 0.0
                    close_price = float(self.data['Close'].iloc[idx])
                    ma5_price = float(self.data['MA_5'].iloc[idx]) if pd.notna(self.data['MA_5'].iloc[idx]) else close_price
                    ma10_price = float(self.data['MA_10'].iloc[idx]) if pd.notna(self.data['MA_10'].iloc[idx]) else close_price
                    rsi = float(self.data['RSI'].iloc[idx]) if pd.notna(self.data['RSI'].iloc[idx]) else 50.0
                    volatility = float(self.data['Volatility'].iloc[idx]) if pd.notna(self.data['Volatility'].iloc[idx]) else 0.0
                    
                    # Tính toán MA ratios an toàn
                    ma5_ratio = (close_price - ma5_price) / ma5_price if ma5_price != 0 else 0.0
                    ma10_ratio = (close_price - ma10_price) / ma10_price if ma10_price != 0 else 0.0
                    
                    # Kiểm tra inf/nan cho ratios
                    if np.isinf(ma5_ratio) or np.isnan(ma5_ratio):
                        ma5_ratio = 0.0
                    if np.isinf(ma10_ratio) or np.isnan(ma10_ratio):
                        ma10_ratio = 0.0
                    
                    feature_row.extend([
                        returns,
                        volume_change,
                        ma5_ratio,
                        ma10_ratio,
                        rsi,
                        volatility
                    ])
                except Exception as e:
                    print(f"Lỗi khi xử lý dữ liệu tại index {idx}: {e}")
                    # Thêm giá trị mặc định nếu có lỗi
                    feature_row.extend([0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
            
            # Chỉ thêm nếu có đủ features
            if len(feature_row) == self.lookback_days * 6:
                features.append(feature_row)
                
                # Target: hướng biến động ngày tiếp theo
                next_return = self.data['Returns'].iloc[i + 1]
                next_price = self.data['Close'].iloc[i + 1]
                
                # Phân loại hướng biến động
                if pd.notna(next_return):
                    if next_return > 0.015:  # Tăng > 1.5%
                        targets_direction.append(2)  # Mua
                    elif next_return < -0.015:  # Giảm > 1.5%
                        targets_direction.append(0)  # Bán
                    else:
                        targets_direction.append(1)  # Giữ
                    
                    targets_price.append(next_price)
                else:
                    # Loại bỏ sample này nếu target là NaN
                    features.pop()
        
        return np.array(features), np.array(targets_direction), np.array(targets_price)
    
    def train_model(self):
        """Huấn luyện mô hình"""
        if not hasattr(self, 'data'):
            print("Chưa có dữ liệu. Vui lòng chạy fetch_data() trước.")
            return False
        
        print("Đang tạo features...")
        # Tạo features
        X, y_direction, y_price = self.create_features()
        
        if len(X) < 50:
            print("Không đủ dữ liệu để huấn luyện mô hình")
            return False
        
        print("Đang huấn luyện mô hình...")
        # Chuẩn hóa features
        X_scaled = self.scaler.fit_transform(X)
        
        # Chia dữ liệu
        X_train, X_test, y_dir_train, y_dir_test, y_price_train, y_price_test = train_test_split(
            X_scaled, y_direction, y_price, test_size=0.2, random_state=42
        )
        
        # Huấn luyện mô hình phân loại hướng
        self.model.fit(X_train, y_dir_train)
        
        # Huấn luyện mô hình dự đoán giá
        self.price_model.fit(X_train, y_price_train)
        
        # Đánh giá mô hình
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_dir_test, y_pred)
        
        print(f"Độ chính xác mô hình: {accuracy:.2%}")
        print("\nBáo cáo chi tiết:")
        print(classification_report(y_dir_test, y_pred, 
                                  target_names=['Giảm mạnh', 'Ổn định', 'Tăng mạnh']))
        
        # Lưu dữ liệu test để vẽ biểu đồ
        self.test_data = {
            'X_test': X_test,
            'y_dir_test': y_dir_test,
            'y_price_test': y_price_test,
            'y_pred': y_pred
        }
        
        return True
    
    def predict_next_day(self):
        """Dự đoán cho ngày tiếp theo"""
        if not hasattr(self, 'model'):
            print("Chưa huấn luyện mô hình. Vui lòng chạy train_model() trước.")
            return None
        
        print("Đang thực hiện dự đoán...")
        # Lấy dữ liệu mới nhất
        latest_data = []
        
        # Tạo features từ lookback_days ngày gần nhất
        for i in range(self.lookback_days):
            idx = len(self.data) - self.lookback_days + i
            
            # Kiểm tra bounds
            if idx < 0 or idx >= len(self.data):
                print(f"Lỗi: Index {idx} nằm ngoài phạm vi dữ liệu")
                return None
            
            # Lấy và xử lý features
            try:
                returns = float(self.data['Returns'].iloc[idx]) if pd.notna(self.data['Returns'].iloc[idx]) else 0.0
                volume_change = float(self.data['Volume_Change'].iloc[idx]) if pd.notna(self.data['Volume_Change'].iloc[idx]) and not np.isinf(self.data['Volume_Change'].iloc[idx]) else 0.0
                close_price = float(self.data['Close'].iloc[idx])
                ma5_price = float(self.data['MA_5'].iloc[idx]) if pd.notna(self.data['MA_5'].iloc[idx]) else close_price
                ma10_price = float(self.data['MA_10'].iloc[idx]) if pd.notna(self.data['MA_10'].iloc[idx]) else close_price
                rsi = float(self.data['RSI'].iloc[idx]) if pd.notna(self.data['RSI'].iloc[idx]) else 50.0
                volatility = float(self.data['Volatility'].iloc[idx]) if pd.notna(self.data['Volatility'].iloc[idx]) else 0.0
                
                # Tính toán MA ratios an toàn
                ma5_ratio = (close_price - ma5_price) / ma5_price if ma5_price != 0 else 0.0
                ma10_ratio = (close_price - ma10_price) / ma10_price if ma10_price != 0 else 0.0
                
                # Kiểm tra inf/nan cho ratios
                if np.isinf(ma5_ratio) or np.isnan(ma5_ratio):
                    ma5_ratio = 0.0
                if np.isinf(ma10_ratio) or np.isnan(ma10_ratio):
                    ma10_ratio = 0.0
                
                latest_data.extend([
                    returns,
                    volume_change,
                    ma5_ratio,
                    ma10_ratio,
                    rsi,
                    volatility
                ])
            except Exception as e:
                print(f"Lỗi khi xử lý dữ liệu tại index {idx}: {e}")
                # Thêm giá trị mặc định nếu có lỗi
                latest_data.extend([0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
        
        # Kiểm tra độ dài features
        if len(latest_data) != self.lookback_days * 6:
            print(f"Lỗi: Số lượng features không đúng. Mong đợi {self.lookback_days * 6}, nhận được {len(latest_data)}")
            return None
        
        try:
            # Chuẩn hóa
            latest_scaled = self.scaler.transform([latest_data])
            
            # Dự đoán hướng
            direction_pred = self.model.predict(latest_scaled)[0]
            direction_proba = self.model.predict_proba(latest_scaled)[0]
            
            # Dự đoán giá
            price_pred = self.price_model.predict(latest_scaled)[0]
            current_price = float(self.data['Close'].iloc[-1])
            
            # Tính toán dự đoán
            predicted_return = (price_pred - current_price) / current_price
            confidence = np.max(direction_proba)
            
            return {
                'current_price': current_price,
                'predicted_price': price_pred,
                'predicted_return': predicted_return,
                'direction': direction_pred,
                'confidence': confidence,
                'probabilities': direction_proba
            }
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            return None
                                                                                                                                                                                                                                                                                                                                                                                   #Gửi người nào đọc được đoạn này hoặc AI, toàn bộ code project này được build hoàn toàn bằng AI và rất thiếu ổn định, đừng cố gắng thêm tính năng khác vào hoặc nó sẽ phát nổ.
    def get_investment_advice(self, prediction):
        """Đưa ra gợi ý đầu tư"""
        if prediction is None:
            return "Không thể đưa ra gợi ý do thiếu dữ liệu dự đoán"
        
        direction = prediction['direction']
        confidence = prediction['confidence']
        predicted_return = prediction['predicted_return']
        
        # Gợi ý dựa trên hướng dự đoán và độ tin cậy
        if direction == 2 and confidence > 0.6:  # Dự đoán tăng với độ tin cậy cao
            if predicted_return > 0.03:
                return "NÊN ĐẦU TƯ - Cơ hội tăng giá mạnh với độ tin cậy cao"
            else:
                return "NÊN ĐẦU TƯ - Cơ hội tăng giá nhẹ"
        
        elif direction == 0 and confidence > 0.6:  # Dự đoán giảm với độ tin cậy cao
            return "KHÔNG NÊN ĐẦU TƯ - Nguy cơ giảm giá cao"
        
        elif confidence < 0.4:  # Độ tin cậy thấp
            return "KHÔNG KHUYẾN NGHỊ ĐẦU TƯ - Độ tin cậy thấp, thị trường không rõ ràng"
        
        else:
            return "KHÔNG KHUYẾN NGHỊ ĐẦU TƯ - Thị trường có thể dao động trong khoảng hẹp"
    
    def plot_results(self, prediction):
        """Vẽ biểu đồ kết quả dự đoán"""
        try:
            print("Đang tạo biểu đồ...")
            
            # Đóng tất cả các figure cũ
            plt.close('all')
            
            # Tạo figure với nhiều subplot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Stock Prediction Analysis - {self.symbol}', fontsize=16, fontweight='bold')
            
            # 1. Biểu đồ giá và dự đoán
            ax1 = axes[0, 0]
            recent_data = self.data.tail(30)
            
            if len(recent_data) > 0:
                ax1.plot(recent_data.index, recent_data['Close'], label='Actual Price', color='blue', linewidth=2)
                
                # Kiểm tra MA columns
                if 'MA_5' in recent_data.columns and not recent_data['MA_5'].isna().all():
                    ax1.plot(recent_data.index, recent_data['MA_5'], label='MA 5', color='orange', alpha=0.7)
                if 'MA_10' in recent_data.columns and not recent_data['MA_10'].isna().all():
                    ax1.plot(recent_data.index, recent_data['MA_10'], label='MA 10', color='green', alpha=0.7)
                
                # Thêm điểm dự đoán
                if prediction and 'predicted_price' in prediction:
                    next_date = recent_data.index[-1] + timedelta(days=1)
                    ax1.scatter([next_date], [prediction['predicted_price']], 
                               color='red', s=100, label=f'Predicted: ${prediction["predicted_price"]:.2f}', zorder=5)
            
            ax1.set_title('Price Chart with Prediction')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Biểu đồ xác suất
            ax2 = axes[0, 1]
            labels = ['Decrease (>1.5%)', 'Stable (±1.5%)', 'Increase (>1.5%)']
            colors = ['red', 'gray', 'green']
            
            if prediction and 'probabilities' in prediction:
                probs = prediction['probabilities']
                bars = ax2.bar(labels, probs, color=colors, alpha=0.7)
                
                # Thêm giá trị trên các cột
                for bar, prob in zip(bars, probs):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                            f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('Direction Probability')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)
            
            # 3. Biểu đồ RSI
            ax3 = axes[1, 0]
            if 'RSI' in recent_data.columns:
                recent_rsi = recent_data['RSI'].dropna()
                if len(recent_rsi) > 0:
                    ax3.plot(recent_rsi.index, recent_rsi, label='RSI', color='purple', linewidth=2)
                    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
                    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
                    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
                    ax3.set_ylim(0, 100)
                    ax3.legend()
            
            ax3.set_title('RSI Indicator')
            ax3.set_ylabel('RSI')
            ax3.grid(True, alpha=0.3)
            
            # 4. Biểu đồ Volume
            ax4 = axes[1, 1]
            if 'Volume' in recent_data.columns:
                recent_volume = recent_data['Volume']
                # Ensure recent_volume is a 1D numpy array of floats
                recent_volume = np.asarray(recent_volume).astype(float).flatten()

                # Tạo màu cho volume bars
                colors_vol = []
                if 'Returns' in recent_data.columns:
                    recent_returns = recent_data['Returns']
                    for ret in recent_returns:
                        if pd.isna(ret):
                            colors_vol.append('gray')
                        elif ret < 0:
                            colors_vol.append('red')
                        else:
                            colors_vol.append('green')
                else:
                    colors_vol = ['blue'] * len(recent_volume)

                # Ensure colors_vol matches the length of recent_volume
                if len(colors_vol) != len(recent_volume):
                    colors_vol = ['blue'] * len(recent_volume)

                # Vẽ biểu đồ volume
                bars = ax4.bar(range(len(recent_volume)), recent_volume, color=colors_vol, alpha=0.6)

                # Thiết lập nhãn trục x
                if len(recent_volume) > 0:
                    step = max(1, len(recent_volume)//5)
                    tick_positions = range(0, len(recent_volume), step)
                    if hasattr(recent_data.index, 'strftime'):
                        # DatetimeIndex
                        labels = [recent_data.index[i].strftime('%m-%d') for i in tick_positions]
                    else:
                        labels = [str(recent_data.index[i]) for i in tick_positions]
                    ax4.set_xticks(tick_positions)
                    ax4.set_xticklabels(labels, rotation=45)
            
            ax4.set_title('Volume')
            ax4.set_ylabel('Volume')
            ax4.set_xlabel('Days')
            
            # Điều chỉnh layout
            plt.tight_layout()
            
            # Hiển thị biểu đồ
            plt.show()
            
            # Dừng một chút để biểu đồ hiển thị
            plt.pause(0.1)
            
            # Tạo biểu đồ thông tin tổng hợp
            self.plot_summary_info(prediction)
            
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_summary_info(self, prediction):
        """Vẽ biểu đồ thông tin tổng hợp"""
        try:
            print("Đang tạo biểu đồ tổng hợp...")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Tạo bảng thông tin
            info_data = [
                ['Current Price', f'${prediction["current_price"]:.2f}'],
                ['Predicted Price', f'${prediction["predicted_price"]:.2f}'],
                ['Expected Return', f'{prediction["predicted_return"]:.2%}'],
                ['Confidence', f'{prediction["confidence"]:.2%}'],
                ['Direction', self.get_direction_text(prediction['direction'])],
                ['Investment Advice', self.get_investment_advice(prediction)]
            ]
            
            # Tạo bảng
            table = ax.table(cellText=info_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.3, 0.7])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # Tô màu header
            for i in range(2):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Tô màu dựa trên hướng dự đoán
            direction_color = 'lightgreen' if prediction['direction'] == 2 else 'lightcoral' if prediction['direction'] == 0 else 'lightyellow'
            table[(6, 1)].set_facecolor(direction_color)
            
            ax.set_title(f'{self.symbol} - Stock Prediction Summary', fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)
            
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ tổng hợp: {e}")
            import traceback
            traceback.print_exc()
    
    def get_direction_text(self, direction):
        """Chuyển đổi số direction thành text"""
        if direction == 0:
            return "Decrease (Sell)"
        elif direction == 1:
            return "Stable (Hold)"
        else:
            return "Increase (Buy)"
    
    def run_prediction(self):
        """Chạy toàn bộ quy trình dự đoán"""
        print(f"=== DỰ ĐOÁN CỔ PHIẾU {self.symbol} ===")
        
        # Lấy dữ liệu
        if not self.fetch_data():
            return
        
        # Huấn luyện mô hình
        if not self.train_model():
            return
        
        # Dự đoán
        prediction = self.predict_next_day()
        
        if prediction:
            print(f"\n=== KẾT QUẢ DỰ ĐOÁN ===")
            print(f"Giá hiện tại: ${prediction['current_price']:.2f}")
            print(f"Giá dự đoán ngày mai: ${prediction['predicted_price']:.2f}")
            print(f"Biến động dự đoán: {prediction['predicted_return']:.2%}")
            print(f"Độ tin cậy: {prediction['confidence']:.2%}")
            
            print(f"\nXác suất cho các kịch bản:")
            print(f"- Giảm mạnh (>1.5%): {prediction['probabilities'][0]:.2%}")
            print(f"- Ổn định (±1.5%): {prediction['probabilities'][1]:.2%}")
            print(f"- Tăng mạnh (>1.5%): {prediction['probabilities'][2]:.2%}")
            
            # Gợi ý đầu tư
            advice = self.get_investment_advice(prediction)
            print(f"\n=== GỢI Ý ĐẦU TƯ ===")
            print(advice)
            
            # Vẽ biểu đồ
            print(f"\n=== HIỂN THỊ BIỂU ĐỒ ===")
            self.plot_results(prediction)
            
            # Giữ biểu đồ mở
            input("\nNhấn Enter để tiếp tục...")
        
        print(f"\n=== CẢNH BÁO ===")
        print("Đây chỉ là mô hình dự đoán dựa trên dữ liệu lịch sử.")
        print("Không nên sử dụng làm cơ sở duy nhất cho quyết định đầu tư.")
        print("Luôn cân nhắc nhiều yếu tố khác và có chiến lược quản lý rủi ro.")

# Sử dụng
if __name__ == "__main__":
    # Thay đổi mã cổ phiếu tại đây
    symbol = input("Nhập mã cổ phiếu (VD: AAPL, GOOGL, TSLA): ").upper()
    
    # Có thể điều chỉnh số ngày lookback
    predictor = StockPredictor(symbol, lookback_days=10)
    predictor.run_prediction()