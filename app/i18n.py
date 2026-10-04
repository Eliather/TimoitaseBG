import json
import os
from pathlib import Path

# Diccionario principal de idiomas
TRANSLATIONS = {
    "es": {
        "lang_name": "Español",
        "btn_open": "Abrir",
        "btn_save": "Guardar",
        "btn_compare": "Comparar",
        "btn_zoom_fit": "Ajustar",
        "btn_kofi": "☕ Apoyar a Eliather",
        "tooltip_open": "Abrir imagen desde el explorador (Ctrl+O) o pegar con Ctrl+V",
        "tooltip_save": "Guardar imagen procesada en disco (Ctrl+S)",
        "tooltip_theme": "Cambiar entre tema Claro/Oscuro",
        "tooltip_lang": "Cambiar idioma / Change language",
        "tooltip_undo": "Deshacer última acción (Ctrl+Z)",
        "tooltip_redo": "Rehacer acción (Ctrl+Y)",
        "tooltip_reset": "Recuperar Original (Restaura la imagen inicial)",
        "tooltip_zoom_out": "Reducir zoom",
        "tooltip_zoom_in": "Aumentar zoom",
        "tooltip_zoom_fit": "Ajustar imagen completa a la ventana",
        "tooltip_zoom_100": "Zoom al 100%",
        "tooltip_compare": "Comparar antes/después con cortina interactiva",
        "tooltip_kofi": "Cómprame un café en Ko-fi para apoyar el desarrollo",
        "tooltip_device": "Haz clic para alternar entre aceleración GPU y modo CPU seguro",
        "status_ready": "Listo. Arrastra una imagen o pulsa 'Abrir'.",
        "status_saved": "Imagen guardada en: {name}",
        "status_undo": "Deshecho: {action}",
        "status_redo": "Rehecho: {action}",
        "status_reset": "Se restauró la imagen original.",
        "status_device_changed": "Modo de dispositivo cambiado a: {device}",
        "popup_save_success_title": "Guardado con éxito",
        "popup_save_success_msg": "La imagen se ha guardado correctamente en:\n{path}",
        "popup_no_image_title": "Sin imagen",
        "popup_no_image_msg": "No hay ninguna imagen activa para guardar.",
        "popup_error_save_title": "Error al guardar",
        "popup_error_save_msg": "No se pudo guardar la imagen:\n{error}",
        "popup_device_title": "Dispositivo Actualizado",
        "popup_device_msg": "El procesamiento ahora se realizará en: {device}",
        "popup_lang_changed_title": "Idioma cambiado",
        "popup_lang_changed_msg": "Reinicia la aplicación para aplicar los cambios de idioma.",
        "btn_ok": "Aceptar",
        "sidebar_title_tools": "Herramientas",
        "sidebar_brush": "Pincel",
        "sidebar_eraser": "Borrador",
        "sidebar_size": "Tamaño",
        "sidebar_smoothness": "Suavizar",
        "sidebar_opacity": "Opacidad",
        "sidebar_hardness": "Dureza",
        "sidebar_action_remove_bg": "Eliminar Fondo (IA)",
        "sidebar_action_restore_bg": "Restaurar Fondo",
        "sidebar_action_autocrop": "Auto Recortar",
        "sidebar_action_apply_bg": "Fondo Sólido",
        "sidebar_action_remove_solid": "Quitar Fondo Sólido",
        "sidebar_action_batch": "Procesar Carpeta",
        "color_picker_title": "Seleccionar Color de Fondo",
        "batch_dialog_title": "Seleccionar Carpeta",
        "batch_status": "Procesadas {current}/{total} imágenes...",
        "batch_finished": "Proceso por lotes finalizado. {count} imágenes guardadas.",
    },
    "en": {
        "lang_name": "English",
        "btn_open": "Open",
        "btn_save": "Save",
        "btn_compare": "Compare",
        "btn_zoom_fit": "Fit",
        "btn_kofi": "☕ Support Eliather",
        "tooltip_open": "Open image from explorer (Ctrl+O) or paste with Ctrl+V",
        "tooltip_save": "Save processed image to disk (Ctrl+S)",
        "tooltip_theme": "Toggle Light/Dark theme",
        "tooltip_lang": "Cambiar idioma / Change language",
        "tooltip_undo": "Undo last action (Ctrl+Z)",
        "tooltip_redo": "Redo action (Ctrl+Y)",
        "tooltip_reset": "Restore Original (Restores the initial image)",
        "tooltip_zoom_out": "Zoom out",
        "tooltip_zoom_in": "Zoom in",
        "tooltip_zoom_fit": "Fit entire image to window",
        "tooltip_zoom_100": "Zoom to 100%",
        "tooltip_compare": "Compare before/after with interactive curtain",
        "tooltip_kofi": "Buy me a coffee on Ko-fi to support development",
        "tooltip_device": "Click to toggle between GPU acceleration and safe CPU mode",
        "status_ready": "Ready. Drag an image or click 'Open'.",
        "status_saved": "Image saved to: {name}",
        "status_undo": "Undone: {action}",
        "status_redo": "Redone: {action}",
        "status_reset": "Original image restored.",
        "status_device_changed": "Device mode changed to: {device}",
        "popup_save_success_title": "Saved Successfully",
        "popup_save_success_msg": "The image was successfully saved to:\n{path}",
        "popup_no_image_title": "No Image",
        "popup_no_image_msg": "There is no active image to save.",
        "popup_error_save_title": "Save Error",
        "popup_error_save_msg": "Could not save the image:\n{error}",
        "popup_device_title": "Device Updated",
        "popup_device_msg": "Processing will now be performed on: {device}",
        "popup_lang_changed_title": "Language changed",
        "popup_lang_changed_msg": "Please restart the application to apply language changes.",
        "btn_ok": "OK",
        "sidebar_title_tools": "Tools",
        "sidebar_brush": "Brush",
        "sidebar_eraser": "Eraser",
        "sidebar_size": "Size",
        "sidebar_smoothness": "Smoothness",
        "sidebar_opacity": "Opacity",
        "sidebar_hardness": "Hardness",
        "sidebar_action_remove_bg": "Remove Background (AI)",
        "sidebar_action_restore_bg": "Restore Background",
        "sidebar_action_autocrop": "Auto Crop",
        "sidebar_action_apply_bg": "Solid Background",
        "sidebar_action_remove_solid": "Remove Solid BG",
        "sidebar_action_batch": "Batch Process Folder",
        "color_picker_title": "Select Background Color",
        "batch_dialog_title": "Select Folder",
        "batch_status": "Processed {current}/{total} images...",
        "batch_finished": "Batch process finished. {count} images saved.",
    },
    "ru": {
        "lang_name": "Русский",
        "btn_open": "Открыть",
        "btn_save": "Сохранить",
        "btn_compare": "Сравнить",
        "btn_zoom_fit": "Вместить",
        "btn_kofi": "☕ Поддержать Eliather",
        "tooltip_open": "Открыть изображение из проводника (Ctrl+O) или вставить через Ctrl+V",
        "tooltip_save": "Сохранить обработанное изображение на диск (Ctrl+S)",
        "tooltip_theme": "Переключить Светлую/Темную тему",
        "tooltip_lang": "Изменить язык / Change language",
        "tooltip_undo": "Отменить последнее действие (Ctrl+Z)",
        "tooltip_redo": "Повторить действие (Ctrl+Y)",
        "tooltip_reset": "Восстановить оригинал (Восстанавливает начальное изображение)",
        "tooltip_zoom_out": "Уменьшить масштаб",
        "tooltip_zoom_in": "Увеличить масштаб",
        "tooltip_zoom_fit": "Вместить изображение в окно целиком",
        "tooltip_zoom_100": "Масштаб 100%",
        "tooltip_compare": "Сравнить до/после с интерактивной шторкой",
        "tooltip_kofi": "Купите мне кофе на Ko-fi, чтобы поддержать разработку",
        "tooltip_device": "Нажмите для переключения между ускорением GPU и безопасным режимом CPU",
        "status_ready": "Готово. Перетащите изображение или нажмите 'Открыть'.",
        "status_saved": "Изображение сохранено в: {name}",
        "status_undo": "Отменено: {action}",
        "status_redo": "Повторено: {action}",
        "status_reset": "Оригинальное изображение восстановлено.",
        "status_device_changed": "Режим устройства изменен на: {device}",
        "popup_save_success_title": "Успешно сохранено",
        "popup_save_success_msg": "Изображение успешно сохранено в:\n{path}",
        "popup_no_image_title": "Нет изображения",
        "popup_no_image_msg": "Нет активного изображения для сохранения.",
        "popup_error_save_title": "Ошибка сохранения",
        "popup_error_save_msg": "Не удалось сохранить изображение:\n{error}",
        "popup_device_title": "Устройство обновлено",
        "popup_device_msg": "Обработка теперь будет выполняться на: {device}",
        "popup_lang_changed_title": "Язык изменен",
        "popup_lang_changed_msg": "Пожалуйста, перезапустите приложение для применения изменений языка.",
        "btn_ok": "ОК",
        "sidebar_title_tools": "Инструменты",
        "sidebar_brush": "Кисть",
        "sidebar_eraser": "Ластик",
        "sidebar_size": "Размер",
        "sidebar_smoothness": "Сглаживание",
        "sidebar_opacity": "Непрозрачность",
        "sidebar_hardness": "Жесткость",
        "sidebar_action_remove_bg": "Удалить фон (ИИ)",
        "sidebar_action_restore_bg": "Восстановить фон",
        "sidebar_action_autocrop": "Автокадрирование",
        "sidebar_action_apply_bg": "Сплошной фон",
        "sidebar_action_remove_solid": "Удалить сплошной фон",
        "sidebar_action_batch": "Пакетная обработка папки",
        "color_picker_title": "Выберите цвет фона",
        "batch_dialog_title": "Выберите папку",
        "batch_status": "Обработано {current}/{total} изображений...",
        "batch_finished": "Пакетная обработка завершена. {count} изображений сохранено.",
    },
    "zh": {
        "lang_name": "中文",
        "btn_open": "打开",
        "btn_save": "保存",
        "btn_compare": "对比",
        "btn_zoom_fit": "适应",
        "btn_kofi": "☕ 支持 Eliather",
        "tooltip_open": "从资源管理器打开图像 (Ctrl+O) 或使用 Ctrl+V 粘贴",
        "tooltip_save": "将处理后的图像保存到磁盘 (Ctrl+S)",
        "tooltip_theme": "切换浅色/深色主题",
        "tooltip_lang": "更改语言 / Change language",
        "tooltip_undo": "撤消上一步操作 (Ctrl+Z)",
        "tooltip_redo": "重做操作 (Ctrl+Y)",
        "tooltip_reset": "恢复原始图像 (恢复初始图像)",
        "tooltip_zoom_out": "缩小",
        "tooltip_zoom_in": "放大",
        "tooltip_zoom_fit": "使整个图像适应窗口",
        "tooltip_zoom_100": "缩放至 100%",
        "tooltip_compare": "使用交互式幕布进行前后对比",
        "tooltip_kofi": "在 Ko-fi 上请我喝杯咖啡以支持开发",
        "tooltip_device": "点击在 GPU 加速和安全的 CPU 模式之间切换",
        "status_ready": "准备就绪。拖动图像或点击“打开”。",
        "status_saved": "图像已保存到：{name}",
        "status_undo": "已撤消：{action}",
        "status_redo": "已重做：{action}",
        "status_reset": "已恢复原始图像。",
        "status_device_changed": "设备模式已更改为：{device}",
        "popup_save_success_title": "保存成功",
        "popup_save_success_msg": "图像已成功保存到：\n{path}",
        "popup_no_image_title": "无图像",
        "popup_no_image_msg": "没有活动的图像可供保存。",
        "popup_error_save_title": "保存错误",
        "popup_error_save_msg": "无法保存图像：\n{error}",
        "popup_device_title": "设备已更新",
        "popup_device_msg": "处理现在将在以下设备上执行：{device}",
        "popup_lang_changed_title": "语言已更改",
        "popup_lang_changed_msg": "请重新启动应用程序以应用语言更改。",
        "btn_ok": "确定",
        "sidebar_title_tools": "工具",
        "sidebar_brush": "画笔",
        "sidebar_eraser": "橡皮擦",
        "sidebar_size": "大小",
        "sidebar_smoothness": "平滑度",
        "sidebar_opacity": "不透明度",
        "sidebar_hardness": "硬度",
        "sidebar_action_remove_bg": "去除背景 (AI)",
        "sidebar_action_restore_bg": "恢复背景",
        "sidebar_action_autocrop": "自动裁剪",
        "sidebar_action_apply_bg": "纯色背景",
        "sidebar_action_remove_solid": "移除纯色背景",
        "sidebar_action_batch": "批量处理文件夹",
        "color_picker_title": "选择背景颜色",
        "batch_dialog_title": "选择文件夹",
        "batch_status": "已处理 {current}/{total} 张图像...",
        "batch_finished": "批处理完成。共保存 {count} 张图像。",
    }
}

class Translator:
    def __init__(self):
        self.current_lang = "es"
        self._load_lang_preference()

    def _load_lang_preference(self):
        pref_file = Path(os.path.expanduser("~")) / ".timoitasebg_lang.json"
        if pref_file.exists():
            try:
                with open(pref_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    lang = data.get("lang", "es")
                    if lang in TRANSLATIONS:
                        self.current_lang = lang
            except:
                pass

    def _save_lang_preference(self):
        pref_file = Path(os.path.expanduser("~")) / ".timoitasebg_lang.json"
        try:
            with open(pref_file, "w", encoding="utf-8") as f:
                json.dump({"lang": self.current_lang}, f)
        except:
            pass

    def set_language(self, lang_code):
        if lang_code in TRANSLATIONS:
            self.current_lang = lang_code
            self._save_lang_preference()

    def get(self, key, **kwargs):
        lang_dict = TRANSLATIONS.get(self.current_lang, TRANSLATIONS["en"])
        text = lang_dict.get(key, TRANSLATIONS["es"].get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text
        return text

# Instancia global
_translator = Translator()

def tr(key, **kwargs):
    return _translator.get(key, **kwargs)

def get_current_language():
    return _translator.current_lang

def set_language(lang_code):
    _translator.set_language(lang_code)

def get_available_languages():
    return [(code, data["lang_name"]) for code, data in TRANSLATIONS.items()]
