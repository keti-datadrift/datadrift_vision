#https://stackoverflow.com/questions/74909513/issue-with-filtering-flowlayout-items-in-pyqt5
import sys, re, os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap,QPainter,QFont
from PyQt5 import uic
# from PyQt5 import QtGui
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QTimer
import time
import socket
import redis
import debugpy
import copy
import traceback
import ctypes

PATTERN = re.compile(r'(\d+)(\.)([0-9a-zA-Z]+$)')
HTML_STRING = '<p align=\"center\">{body}</p>'
NUM_THUMBS_TO_DEL = 10000
image_label_width = 16*40
image_label_height = 9*40
name_label_min_width = 120
name_label_max_width = 16*40
margin_hrz = 35
margin_vrt = 20
max_width = 16*40+margin_hrz*2
max_height = 9*40+10*30-100+20+100+170

class RealTimeStr(QThread):
    def __init__(self, signal_realtime=None, msg_=None, parent=None):
        super().__init__()
        msg = msg_.split('^')
        self.aimemo = msg[0]
        self.hashkey = msg[1]

        self.signal_realtime = signal_realtime
    def run(self):
        words = self.aimemo.split(' ')
        aimemo_sub = ''
        for word in words:
            aimemo_sub += word + ' '
            msg_sub = f'{aimemo_sub}^{self.hashkey}'
            self.signal_realtime.emit(msg_sub)
            time.sleep(0.1)        

        self.quit()

class Worker(QThread):
    finished = pyqtSignal(str)
    signal_realtime = pyqtSignal(str)
    def __init__(self, sec=0, parent=None):
        super().__init__()
        self.user_r = redis.Redis(host='127.0.0.1', port=6379, db=0)
        # self.user_r = redis.Redis(host='localhost', port=6379, db=0)
        self.user_pub_sub = self.user_r.pubsub()
        self.user_pub_sub.subscribe('aimemo_server')
        self.msg = None

    def run(self):
        # Listen for messages
        i = 0
        for message in self.user_pub_sub.listen():
            if message['type'] == 'message':
                self.msg = message['data'].decode()
                msg = self.msg.split('^')
                fname = msg[0]
                eventname = msg[1]                 
                score = msg[2]
                aimemo = msg[3].replace('2.','\n2.').replace('3.','\n3.').replace('4.','\n4.').replace('5.','\n5.').strip()
                camera_uid = msg[4]
                
                hashkey = f'rthread{i}'
                msg_sub = f"{fname}^{eventname}^{score}^{aimemo}^{hashkey}^{camera_uid}"
                self.finished.emit(msg_sub)
                time.sleep(0.01)
                msg = f'{aimemo}^{hashkey}'
                rt_str = RealTimeStr(self.signal_realtime,msg)
                rt_str.start()
                i+=1

            time.sleep(0.01)

class FlowLayout(QtWidgets.QLayout):
    heightChanged = QtCore.pyqtSignal(int)
    scrollMax = QtCore.pyqtSignal(int)
    def __init__(self, parent=None):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setContentsMargins(QtCore.QMargins(0, 0, 0, 0))
        self.NUM_THUMB_LINE = 2
        self._item_list = []
        self._active_item_list = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]

        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)

        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._do_layout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.update()
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()

        for item in self._active_item_list:
            size = size.expandedTo(item.minimumSize())

        size += QtCore.QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    # def _do_layout(self, rect, test_only):
    #     x = rect.x()
    #     y = rect.y() + max_height
    #     line_height = 0
    #     spacing = self.spacing()
    #     cnt = 0
    #     for item in self._item_list:
    #         style = item.widget().style()
    #         layout_spacing_x = style.layoutSpacing(
    #             QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal
    #         )
    #         layout_spacing_y = style.layoutSpacing(
    #             QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical
    #         )
    #         space_x = spacing + layout_spacing_x
    #         space_y = spacing + layout_spacing_y
    #         next_x = x + item.sizeHint().width() + space_x
            
    #         cnt += 1
    #         print('cnt on line',cnt)

    #         # print(item.sizeHint().width(),item.sizeHint().height())
    #         if next_x - space_x > rect.right() and line_height > 0:
    #             print('---------cnt newline',cnt)
    #             cnt = 0
    #             x = rect.x()
    #             y = y + line_height + space_y
    #             next_x = x + item.sizeHint().width() + space_x
    #             line_height = 0
    #         if not test_only:
    #             item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y-max_height), item.sizeHint()))

    #         x = next_x
    #         line_height = max(line_height, item.sizeHint().height())

    #     new_height = y + line_height - rect.y()
    #     self.heightChanged.emit(new_height)
    #     return new_height 
    def change_NUM_THUMB_LINE(self, NUM_THUMB_LINE):
        self.NUM_THUMB_LINE = NUM_THUMB_LINE

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()
        # print('_do_layout.NUM_THUMB_LINE=',self.NUM_THUMB_LINE)

        for idx,item in enumerate(self._item_list):
            style = item.widget().style()
            layout_spacing_x = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Horizontal
            )
            layout_spacing_y = style.layoutSpacing(
                QtWidgets.QSizePolicy.PushButton, QtWidgets.QSizePolicy.PushButton, QtCore.Qt.Vertical
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x

            if (idx+1)%self.NUM_THUMB_LINE==0 and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        if len(self._item_list)<=self.NUM_THUMB_LINE:
            line_height = max_height+margin_vrt
        new_height = y + line_height - rect.y()
        self.heightChanged.emit(new_height)
        self.scrollMax.emit(new_height)
        return new_height 
    
class WordwrapLabel(QtWidgets.QTextEdit):
    size_change_signal = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        # Call the parent constructor
        super(WordwrapLabel, self).__init__(parent)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        self.setStyleSheet('background:transparent;')
        self.selected = False
        self.default_style_sheet = self.styleSheet()
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setReadOnly(True)

        self.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    def wheelEvent(self, event):
        event.ignore()
        super(WordwrapLabel, self).wheelEvent(event)
    def resizeEvent(self, event):
        super(WordwrapLabel, self).resizeEvent(event)
        font = self.document().defaultFont()
        font_metrics = QtGui.QFontMetrics(font)
        text_size = font_metrics.size(0, self.toPlainText())
        if text_size.width() >= self.size().width():
            line_factor = int(text_size.width() / self.size().width()) + 1
            new_height = (text_size.height() * line_factor) + 15
            self.setMinimumHeight(new_height)
            self.setMaximumHeight(new_height)
        else:
            height = text_size.height() + 15
            self.setMinimumHeight(height)
            self.setMaximumHeight(height)
class Thumbnail(QtWidgets.QWidget):
    def __init__(self, parent=None, image_path=None,eventname=None,score=None,aimemo=None,camera_uid=None):
        # Call the parent constructor
        super(Thumbnail, self).__init__(parent)
        self.image_path=image_path
        self.eventname=eventname
        self.score=score
        self.camera_uid=camera_uid

        self.image_label_width = 0
        self.image_label_height = 0
        self.name_label_min_width = 0
        self.name_label_max_width = 0
        self.name_label_min_height = 18
        self.max_width = 0
        self.max_height = 0
        self.icon_sizes = ['small', 'medium', 'large']
        self.icon_size_index = 1
        self.current_icon_size = 'medium'
        
        self.widget = QtWidgets.QWidget()
        self.widget.setObjectName('thumbWidget')
        # Apply the stylesheet to the main_widget
        self.widget.setStyleSheet("""
            #thumbWidget {
                border: 2px solid #2B2B2B;  /* Border color and width */
                border-radius: 10px;    /* Optional: rounded corners */
            }
        """)
        master_lyt = QtWidgets.QVBoxLayout()
        sub_lyt = QtWidgets.QVBoxLayout()

        self.image_label = QtWidgets.QLabel('image',self.widget)
        self.image_label.setGeometry(margin_hrz, 0+margin_vrt, image_label_width, image_label_height)
        self.pic = QtGui.QPixmap()
        try:
            self.pic.load(image_path)
        except Exception as e:
            traceback.print_exc()

        self.image_label.setPixmap(self.pic)
        self.image_label.setScaledContents(True)
        self.image_label.setMargin(0)

        self.infos_lbl = QtWidgets.QLabel('infos_lbl',self.widget)
        self.infos_lbl.setGeometry(margin_hrz, image_label_height+5+margin_vrt, image_label_width, 20*3+20)        
        info_text = f'카메라: {camera_uid}\n이벤트: {eventname}\n스코어: {score}'
        self.infos_lbl.setFont(QtGui.QFont("맑은 고딕",20))
        self.infos_lbl.setStyleSheet(f"color: white; font-size: 18px; font-weight: bold;")  
        self.infos_lbl.setText(info_text)
        self.infos_lbl.setMargin(0)
        self.infos_lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)         

        self.aimemo_lbl = QtWidgets.QLabel('aimemo_lbl',self.widget)
        self.aimemo_lbl.setGeometry(margin_hrz, image_label_height+20*3+40+margin_vrt, image_label_width, max_height-(image_label_height+20*3+100))          
        self.aimemo_lbl.setFont(QtGui.QFont("맑은 고딕",20))
        self.aimemo_lbl.setStyleSheet(f"color: white; font-size: 18px; font-weight: bold;")        
        self.context = f'{aimemo}'
        self.aimemo_lbl.setWordWrap(True)
        self.aimemo_lbl.setText(self.context)
        self.aimemo_lbl.setMargin(0)
        self.aimemo_lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft) 

        master_lyt.addWidget(self.widget)

        self.setLayout(master_lyt)

        self.setIconSize(self.icon_sizes[self.icon_size_index])

    def thumbnail_aimemo_update(self, aimemo_sub):
        self.context = f'{aimemo_sub}'
        self.aimemo_lbl.setText(self.context)


    def setIconSize(self, size):

        self.image_label_width = image_label_width
        self.image_label_height = image_label_height
        self.name_label_min_width = name_label_min_width
        self.name_label_max_width = name_label_max_width
        self.max_width = max_width
        self.max_height = max_height

        self.image_label.setMinimumHeight(self.image_label_height)
        self.image_label.setMinimumWidth(self.image_label_width)
        self.image_label.setMaximumHeight(self.image_label_height)
        self.image_label.setMaximumWidth(self.image_label_width)

        self.widget.setMinimumWidth(self.max_width)
        self.widget.setMaximumWidth(self.max_width)
        self.widget.setMinimumHeight(self.max_height)
        self.widget.setMaximumHeight(self.max_height)

class DialogTest(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        # Call the parent constructor
        super(DialogTest, self).__init__(parent)
        # thread 추가
        self.worker = Worker()
        self.worker.finished.connect(self.init_aimemo)
        self.worker.signal_realtime.connect(self.update_aimemo)
        self.worker.start()

        self.my_ip_addr = socket.gethostbyname(socket.getfqdn())
        print(f'my ip_port={self.my_ip_addr}')
        self.verticalScrollBar().setStyleSheet("""
                                                QScrollBar:horizontal {
                                                    min-width: 240px;
                                                    height: 10px;
                                                    background: rgb(80,80,80);
                                                }

                                                QScrollBar:vertical {
                                                    min-height: 240px;
                                                    width: 10px;
                                                    background: rgb(80,80,80);
                                                }

                                                QScrollBar::groove {
                                                    background: black;
                                                    border-radius: 5px;
                                                }

                                                QScrollBar::handle {
                                                    background: white;
                                                    border-radius: 5px;
                                                }

                                                QScrollBar::handle:horizontal {
                                                    width: 25px;
                                                }

                                                QScrollBar::handle:vertical {
                                                    height: 25px;
                                                }
                                               QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                                                    background: none;
                                                }
                                               """)
        self.setWidgetResizable(True)
        # Set the title of the window
        self.setWindowTitle("AI Memo")
        self.selected_icon = None
        # Set the geometry for the window
        self.resize(600, 400)

        self.container = QtWidgets.QWidget()
        self.container_lyt = QtWidgets.QVBoxLayout()
        self.widget = QtWidgets.QWidget()
        self.widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.widget_flow_lyt = FlowLayout()
        self.widget.setMinimumHeight(max_height)
        self.widget_flow_lyt.heightChanged.connect(self.widget.setMinimumHeight)
        self.widget_flow_lyt.heightChanged.emit(max_height)
        self.widget_flow_lyt.scrollMax.connect(self.scroll_to_bottom)        
        self.widget.setLayout(self.widget_flow_lyt)

        self.button_dict = {}
        self.thumbnail_idx_dict = {}        

        self.container_lyt.addWidget(self.widget)
        self.container_lyt.addStretch()
        self.container.setLayout(self.container_lyt)

        selected_color = QColor(0,0,0)       
        self.container.setStyleSheet("background-color: %s;" %selected_color.name())
        self.setWindowIcon(QtGui.QIcon('ivxlogo.png'))
        self.setWidgetResizable(True)
        self.setWidget(self.container)
        self.thumbnail_num = 0


    @pyqtSlot(str)
    def update_aimemo(self, msg_):
        if msg_ is None:
            print('update_aimemo msg_ is None!!!')
            return
        msg = msg_.split('^')
        aimemo_sub = msg[0]
        hashkey = msg[1]
        if hashkey is None:
            print('update_aimemo hashkey is None!!!->return')
            return        
        if hashkey not in self.thumbnail_idx_dict:
            # print(f'hashkey={hashkey} is not in self.thumbnail_idx_dict!!!')
            return
        idx = self.thumbnail_idx_dict[hashkey]
        wgt_last = self.widget_flow_lyt.itemAt(int(idx))
        if wgt_last is None:
            print(self.thumbnail_idx_dict)
            print(f'update_aimemo: at idx={idx}, thumb is None -> return')
            return
        thumb = wgt_last.widget()        
        thumb.thumbnail_aimemo_update(aimemo_sub)

        self.update()
        self.repaint()
    @pyqtSlot(str)
    def init_aimemo(self, msg):
        cnt = self.widget_flow_lyt.count()
        if cnt>NUM_THUMBS_TO_DEL:
            self.widget_flow_lyt.__del__()
            self.thumbnail_idx_dict = {}
            self.thumbnail_num = 0
            self.repaint()
            self.update()
            return
        msg = msg.split('^')

        fname = msg[0]
        eventname = msg[1]                 
        score = msg[2]
        aimemo = msg[3]
        hashkey = msg[4]
        camera_uid = msg[5]

        desc = aimemo

        thumbnail = Thumbnail(self, image_path=fname, eventname=eventname,score=score,aimemo='*',camera_uid=camera_uid)

        self.widget_flow_lyt.addWidget(thumbnail)
        self.button_dict[desc] = thumbnail
        self.thumbnail_idx_dict[hashkey] = self.thumbnail_num
        self.thumbnail_num += 1

        self.scroll_to_bottom() 
        self.repaint()
        self.resizeEvent(QtGui.QResizeEvent(self.size(), QtCore.QSize()))

    def get_widget_from_layout(self, index):
        # Get the widget from the layout at the given index
        widget = None
        layout_item = self.widget_flow_lyt.itemAt(index)
        if layout_item is not None:
            widget = layout_item.widget()
            if widget is not None:
                print(f"Widget at index {index}: {widget}")
        return widget
    def scroll_to_bottom(self):
        # Scroll to the bottom of the QScrollArea
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    def resizeEvent(self, event):
        super(DialogTest, self).resizeEvent(event)
        width_container = self.container.width()-50
        NUM_THUMB_LINE = int((width_container)/max_width)
        if NUM_THUMB_LINE < 1:
            NUM_THUMB_LINE = 1
        self.widget_flow_lyt.change_NUM_THUMB_LINE(NUM_THUMB_LINE)

    def mousePressEvent(self, event):
        cursor = event.globalPos()
        if event.button() == QtCore.Qt.LeftButton:
            self.highlightIcon(event, cursor)

    def mouseDoubleClickEvent(self, event):
        pos = self.geometry()
        cursor = event.globalPos()
        if event.button() == QtCore.Qt.LeftButton:
            self.highlightIcon(event, cursor)

    def increaseIcon(self):
        for button_name, button_obj in self.button_dict.items():
            button_obj.increaseIcon()

    def decreaseIcon(self):
        for button_name, button_obj in self.button_dict.items():
            button_obj.decreaseIcon()

    def highlightIcon(self, event, cursor):
            clicked_widget = QtWidgets.QApplication.widgetAt(cursor)
            if clicked_widget == self:
                return                
            thumbnail_widget = self.getThumbnailWidget(clicked_widget)
            if not thumbnail_widget:
                return
            if thumbnail_widget == self.selected_icon and event.modifiers() & QtCore.Qt.ControlModifier:
                self.selected_icon.widget.setStyleSheet('')
                self.selected_icon = None
                return
            if self.selected_icon:
                self.selected_icon.widget.setStyleSheet('')
            thumbnail_widget.widget.setStyleSheet('background-color: gray;')
            self.selected_icon = thumbnail_widget

    def layout_widgets(self, layout):
        return (layout.itemAt(i) for i in reversed(range(layout.count())))

    def updateFlowLayout(self):
        for btn_label in self.button_dict:
            if self.button_dict[btn_label] not in [x.widget() for x in self.layout_widgets(self.widget_flow_lyt)]:
                self.widget_flow_lyt.addWidget(self.button_dict[btn_label])
                self.button_dict[btn_label].show()
        for item in self.layout_widgets(self.widget_flow_lyt):
            if not item:
                continue
            widget = item.widget()
        self.resizeEvent(QtGui.QResizeEvent(self.size(), QtCore.QSize()))

    def getThumbnailWidget(self, inputWidget):
        if type(inputWidget) == Thumbnail:
            return inputWidget
        parent_widget = inputWidget.parentWidget()
        thumbnail_found = False
        while parent_widget:
            if type(parent_widget) == Thumbnail:
                thumbnail_found = True
                break
            parent_widget = parent_widget.parentWidget()
        if not thumbnail_found:
            parent_widget = None
        return parent_widget
    
class LayoutExample(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Memo")
        self.setWindowIcon(QtGui.QIcon('ivxlogo.png'))
        self.setStyleSheet('background-color: black;')
        self.title_lbl = QtWidgets.QLabel()
        self.title_lbl.setText('GenAI 기반 실시간 분석 보고')
        self.title_lbl.setFont(QtGui.QFont("맑은 고딕",20))
        self.title_lbl.setStyleSheet(f"color: white; font-size: 60px; font-weight: bold;")
        self.title_lbl.setMinimumHeight = 50
        
        # Create a horizontal layout
        self.main_layout = QtWidgets.QVBoxLayout()
        
        self.main_layout.addWidget(self.title_lbl,alignment=QtCore.Qt.AlignTop | Qt.AlignHCenter) 

        self.sub_layout = QVBoxLayout()
        self.dialog = DialogTest()
        self.dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dialog.setMinimumHeight = 1000
        self.sub_layout.addWidget(self.dialog)
        # Add some buttons to the layout

        self.sub_widget = QtWidgets.QWidget()
        self.sub_widget.setLayout(self.sub_layout)
        self.sub_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_widget.showMaximized()

        self.main_layout.addWidget(self.sub_widget)
        # Set the layout for the parent widget
        self.setLayout(self.main_layout)
        self.setGeometry(0,0,1920,1080)
        # self.adjustSize()
        self.change_titlebar_color()

    def change_titlebar_color(self):
        hwnd = self.winId()  # Get the window ID (sip.voidptr)
        hwnd = int(hwnd)  # Convert to int for ctypes compatibility
        # Constants from Windows API for DwmSetWindowAttribute
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20  # The attribute to change titlebar color

        # Enable dark mode for the title bar (or change color as needed)
        dark_mode_enabled = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(dark_mode_enabled),
            ctypes.sizeof(dark_mode_enabled)
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = LayoutExample()
    w.show()
    sys.exit(app.exec_())