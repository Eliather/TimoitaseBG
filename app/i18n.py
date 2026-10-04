import json
import os
from pathlib import Path

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
        "tooltip_lang": "Cambiar idioma",
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
        
        "brush_settings": "Ajustes del Pincel",
        "lbl_size_thick": "Grosor:",
        "lbl_opacity": "Opacidad:",
        "header_batch": "Procesamiento por Lotes:",
        "batch_desc": "Quita el fondo a múltiples imágenes automáticamente.",
        "btn_batch_input": "📂 Seleccionar carpeta origen...",
        "batch_no_folder": "No hay carpeta seleccionada",
        "btn_batch_output": "📁 Seleccionar carpeta destino...",
        "lbl_batch_model": "Modelo a usar:",
        "chk_batch_clothing": "🛡️ Detector avanzado de ropa",
        "btn_run_batch": "🚀 Iniciar Proceso en Lote",
        "batch_input_path": "Origen: {path}",
        "batch_output_path": "Destino: {path}",
        
        "model_isnet-anime": "IS-Net Anime (Óptimo para Anime, Manga e Ilustración)",
        "model_u2net_human_seg": "U2-Net Human (Detector de Personas y Ropa — Especial Playeras Blancas)",
        "model_birefnet-general": "BiRefNet (Alta Definición General y Bordes Complejos)",
        "model_inspyrenet": "InSPyReNet Swin-B (Fotografía y Retratos Reales — Anti-aliasing Subpíxel)",
        
        "scale_x2": "Escalar 2x",
        "scale_x4": "Escalar 4x (Máxima resolución)",
        "scale_enhance_only": "Solo nitidez (Sin cambiar tamaño)",

        "btn_ok": "Aceptar",
        
        "canvas_drop_title": "Arrastra una imagen aquí",
        "canvas_drop_subtitle": "o usa el botón 'Abrir Imagen' o pega con Ctrl+V",
        "canvas_drop_formats": "Soporta PNG, JPG, JPEG, WEBP y BMP",
        
        "sidebar_title": "Herramientas",
        "sidebar_subtitle": "Edición y mejora con IA local",
        "sidebar_tab_bg": "⬚\nQuitar",
        "sidebar_tab_restore": "✦\nEscalar",
        "sidebar_tab_brush": "🖌\nPincel",
        "sidebar_tab_batch": "📂\nLotes",
        "sidebar_shortcuts_col": "▸ Atajos rápidos",
        "sidebar_shortcuts_exp": "▾ Atajos rápidos",
        "shortcut_zoom": "Rueda",
        "shortcut_zoom_desc": "Zoom in / out",
        "shortcut_pan": "Click medio / Espacio",
        "shortcut_pan_desc": "Arrastrar y mover",
        "shortcut_undo": "Ctrl+Z / Ctrl+Y",
        "shortcut_undo_desc": "Deshacer / Rehacer",
        
        "header_model": "Modelo de segmentación:",
        "model_desc": "Elimina el fondo dejando transparencia limpia optimizada para ilustraciones y anime.",
        "chk_clothing": "🛡️ Detector avanzado de ropa (evita agujeros en prendas blancas)",
        "chk_clothing_tooltip": "Combina la segmentación con un detector semántico de personas y prendas para proteger playeras, camisas y mangas blancas contra fondos claros.",
        "btn_remove_now": "✨ Quitar Fondo Ahora",
        
        "header_crop": "Ajuste de Lienzo:",
        "btn_autocrop_now": "⛶ Recortar al Contenido",
        "btn_autocrop_tooltip": "Aplica primero Quitar Fondo",
        
        "header_bg": "Fondo:",
        "swatch_transparent": "Transparente / Ninguno (Restaurar canal alfa)",
        "swatch_white": "Fondo Blanco (#FFFFFF)",
        "swatch_black": "Fondo Negro (#000000)",
        "swatch_gray": "Fondo Gris (#E2E8F0)",
        "swatch_custom": "Elegir color personalizado...",
        "color_white": "Blanco (#FFFFFF)",
        "color_black": "Negro (#000000)",
        "color_gray": "Gris (#E2E8F0)",
        
        "header_scale": "Factor de mejora:",
        "scale_desc": "Mejora la nitidez y resolución usando Real-ESRGAN especializado en arte e ilustración.",
        "chk_tiling": "Procesar por bloques (Tiling inteligente)",
        "chk_tiling_tooltip": "Evita problemas de memoria en imágenes de alta resolución.",
        "btn_scale_now": "🚀 Escalar Imagen",
        
        "header_brush": "Pincel Mágico:",
        "brush_desc": "Pinta sobre cualquier elemento u objeto que desees eliminar y la IA lo rellenará de forma coherente.",
        "btn_brush_mode": "🖌️ Modo Pincel",
        "btn_paint": "🖌️ Pintar",
        "btn_erase": "🧽 Borrador",
        "lbl_size": "Tamaño:",
        "lbl_smoothness": "Suavizar:",
        "lbl_hardness": "Dureza:",
        "btn_clear_mask": "🗑️ Limpiar Trazo",
        "btn_apply_inpaint": "✨ Aplicar Relleno IA",
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
        "tooltip_lang": "Change language",
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
        
        "brush_settings": "Brush Settings",
        "lbl_size_thick": "Thickness:",
        "lbl_opacity": "Opacity:",
        "header_batch": "Batch Processing:",
        "batch_desc": "Remove background from multiple images automatically.",
        "btn_batch_input": "📂 Select source folder...",
        "batch_no_folder": "No folder selected",
        "btn_batch_output": "📁 Select destination folder...",
        "lbl_batch_model": "Model to use:",
        "chk_batch_clothing": "🛡️ Advanced clothing detector",
        "btn_run_batch": "🚀 Start Batch Process",
        "batch_input_path": "Source: {path}",
        "batch_output_path": "Destination: {path}",
        
        "model_isnet-anime": "IS-Net Anime (Optimal for Anime, Manga and Illustration)",
        "model_u2net_human_seg": "U2-Net Human (Person & Clothing Detector — White Shirts)",
        "model_birefnet-general": "BiRefNet (General High Definition and Complex Edges)",
        "model_inspyrenet": "InSPyReNet Swin-B (Real Photography & Portraits — Subpixel Anti-aliasing)",
        
        "scale_x2": "Upscale 2x",
        "scale_x4": "Upscale 4x (Maximum resolution)",
        "scale_enhance_only": "Sharpen only (No resize)",

        "btn_ok": "OK",
        
        "canvas_drop_title": "Drag an image here",
        "canvas_drop_subtitle": "or use 'Open' button or paste with Ctrl+V",
        "canvas_drop_formats": "Supports PNG, JPG, JPEG, WEBP and BMP",
        
        "sidebar_title": "Tools",
        "sidebar_subtitle": "Local AI editing and enhancement",
        "sidebar_tab_bg": "⬚\nRemove BG",
        "sidebar_tab_restore": "✦\nUpscale",
        "sidebar_tab_brush": "🖌\nBrush",
        "sidebar_tab_batch": "📂\nBatch",
        "sidebar_shortcuts_col": "▸ Quick shortcuts",
        "sidebar_shortcuts_exp": "▾ Quick shortcuts",
        "shortcut_zoom": "Scroll Wheel",
        "shortcut_zoom_desc": "Zoom in / out",
        "shortcut_pan": "Mid Click / Space",
        "shortcut_pan_desc": "Pan and move",
        "shortcut_undo": "Ctrl+Z / Ctrl+Y",
        "shortcut_undo_desc": "Undo / Redo",
        
        "header_model": "Segmentation Model:",
        "model_desc": "Removes background leaving clean transparency optimized for anime and illustrations.",
        "chk_clothing": "🛡️ Advanced clothing detector (prevents holes in white clothes)",
        "chk_clothing_tooltip": "Combines segmentation with semantic detection to protect white shirts and sleeves against bright backgrounds.",
        "btn_remove_now": "✨ Remove Background Now",
        
        "header_crop": "Canvas Fit:",
        "btn_autocrop_now": "⛶ Crop to Content",
        "btn_autocrop_tooltip": "Apply Remove Background first",
        
        "header_bg": "Background:",
        "swatch_transparent": "Transparent / None (Restore Alpha)",
        "swatch_white": "White Background (#FFFFFF)",
        "swatch_black": "Black Background (#000000)",
        "swatch_gray": "Gray Background (#E2E8F0)",
        "swatch_custom": "Choose custom color...",
        "color_white": "White (#FFFFFF)",
        "color_black": "Black (#000000)",
        "color_gray": "Gray (#E2E8F0)",
        
        "header_scale": "Enhancement Factor:",
        "scale_desc": "Improves sharpness and resolution using Real-ESRGAN specialized in art.",
        "chk_tiling": "Process by blocks (Smart Tiling)",
        "chk_tiling_tooltip": "Prevents memory issues on high-res images.",
        "btn_scale_now": "🚀 Upscale Image",
        
        "header_brush": "Magic Brush:",
        "brush_desc": "Paint over any object you want to remove and the AI will cohesively inpaint it.",
        "btn_brush_mode": "🖌️ Brush Mode",
        "btn_paint": "🖌️ Paint",
        "btn_erase": "🧽 Eraser",
        "lbl_size": "Size:",
        "lbl_smoothness": "Smooth:",
        "lbl_hardness": "Hardness:",
        "btn_clear_mask": "🗑️ Clear Stroke",
        "btn_apply_inpaint": "✨ Apply AI Fill",
    },
    "ru": {
        "lang_name": "Русский",
        "btn_open": "Открыть",
        "btn_save": "Сохранить",
        "btn_compare": "Сравнить",
        "btn_zoom_fit": "Вместить",
        "btn_kofi": "☕ Поддержать Eliather",
        "tooltip_open": "Открыть из проводника (Ctrl+O) или вставить через Ctrl+V",
        "tooltip_save": "Сохранить на диск (Ctrl+S)",
        "tooltip_theme": "Переключить тему",
        "tooltip_lang": "Изменить язык",
        "tooltip_undo": "Отменить (Ctrl+Z)",
        "tooltip_redo": "Повторить (Ctrl+Y)",
        "tooltip_reset": "Восстановить оригинал",
        "tooltip_zoom_out": "Уменьшить",
        "tooltip_zoom_in": "Увеличить",
        "tooltip_zoom_fit": "Вместить в окно",
        "tooltip_zoom_100": "Масштаб 100%",
        "tooltip_compare": "Интерактивное сравнение до/после",
        "tooltip_kofi": "Поддержите разработку на Ko-fi",
        "tooltip_device": "Переключение GPU / CPU",
        "status_ready": "Готово. Перетащите изображение или нажмите 'Открыть'.",
        "status_saved": "Сохранено в: {name}",
        "status_undo": "Отменено: {action}",
        "status_redo": "Повторено: {action}",
        "status_reset": "Оригинал восстановлен.",
        "status_device_changed": "Режим изменен на: {device}",
        "popup_save_success_title": "Успешно",
        "popup_save_success_msg": "Сохранено в:\n{path}",
        "popup_no_image_title": "Нет изображения",
        "popup_no_image_msg": "Нет изображения для сохранения.",
        "popup_error_save_title": "Ошибка",
        "popup_error_save_msg": "Не удалось сохранить:\n{error}",
        "popup_device_title": "Устройство обновлено",
        "popup_device_msg": "Обработка на: {device}",
        "popup_lang_changed_title": "Язык изменен",
        "popup_lang_changed_msg": "Перезапустите приложение.",
        
        "brush_settings": "Настройки кисти",
        "lbl_size_thick": "Толщина:",
        "lbl_opacity": "Непрозрачность:",
        "header_batch": "Пакетная обработка:",
        "batch_desc": "Автоматическое удаление фона с нескольких изображений.",
        "btn_batch_input": "📂 Выбрать исходную папку...",
        "batch_no_folder": "Папка не выбрана",
        "btn_batch_output": "📁 Выбрать папку назначения...",
        "lbl_batch_model": "Модель:",
        "chk_batch_clothing": "🛡️ Детектор одежды",
        "btn_run_batch": "🚀 Начать пакетную обработку",
        "batch_input_path": "Источник: {path}",
        "batch_output_path": "Назначение: {path}",
        
        "model_isnet-anime": "IS-Net Anime (Для аниме и иллюстраций)",
        "model_u2net_human_seg": "U2-Net Human (Для людей и белой одежды)",
        "model_birefnet-general": "BiRefNet (Высокая детализация сложных краев)",
        "model_inspyrenet": "InSPyReNet Swin-B (Для фото и портретов)",
        
        "scale_x2": "Увеличить 2x",
        "scale_x4": "Увеличить 4x (Макс. разрешение)",
        "scale_enhance_only": "Только резкость (Без изменения размера)",

        "btn_ok": "ОК",
        
        "canvas_drop_title": "Перетащите изображение сюда",
        "canvas_drop_subtitle": "или используйте кнопку 'Открыть' / Ctrl+V",
        "canvas_drop_formats": "Поддерживает PNG, JPG, JPEG, WEBP и BMP",
        
        "sidebar_title": "Инструменты",
        "sidebar_subtitle": "Редактирование и ИИ-улучшение",
        "sidebar_tab_bg": "⬚\nФон",
        "sidebar_tab_restore": "✦\nУвеличить",
        "sidebar_tab_brush": "🖌\nКисть",
        "sidebar_tab_batch": "📂\nПакет",
        "sidebar_shortcuts_col": "▸ Горячие клавиши",
        "sidebar_shortcuts_exp": "▾ Горячие клавиши",
        "shortcut_zoom": "Колесико мыши",
        "shortcut_zoom_desc": "Масштаб +/-",
        "shortcut_pan": "Ср. клик / Пробел",
        "shortcut_pan_desc": "Панорамирование",
        "shortcut_undo": "Ctrl+Z / Ctrl+Y",
        "shortcut_undo_desc": "Отмена / Повтор",
        
        "header_model": "Модель сегментации:",
        "model_desc": "Удаляет фон, оставляя чистую прозрачность (оптимизировано для аниме).",
        "chk_clothing": "🛡️ Защита белой одежды от удаления",
        "chk_clothing_tooltip": "Семантический детектор одежды для предотвращения дыр на белых рубашках.",
        "btn_remove_now": "✨ Удалить фон",
        
        "header_crop": "Кадрирование:",
        "btn_autocrop_now": "⛶ Кадрировать по контуру",
        "btn_autocrop_tooltip": "Сначала удалите фон",
        
        "header_bg": "Фон:",
        "swatch_transparent": "Прозрачный",
        "swatch_white": "Белый (#FFFFFF)",
        "swatch_black": "Черный (#000000)",
        "swatch_gray": "Серый (#E2E8F0)",
        "swatch_custom": "Выбрать цвет...",
        "color_white": "Белый (#FFFFFF)",
        "color_black": "Черный (#000000)",
        "color_gray": "Серый (#E2E8F0)",
        
        "header_scale": "Фактор улучшения:",
        "scale_desc": "Повышает резкость с помощью Real-ESRGAN.",
        "chk_tiling": "Обработка блоками (Tiling)",
        "chk_tiling_tooltip": "Предотвращает ошибки памяти на больших фото.",
        "btn_scale_now": "🚀 Увеличить",
        
        "header_brush": "Волшебная кисть:",
        "brush_desc": "Закрасьте нежелательные объекты, и ИИ удалит их.",
        "btn_brush_mode": "🖌️ Режим кисти",
        "btn_paint": "🖌️ Рисовать",
        "btn_erase": "🧽 Ластик",
        "lbl_size": "Размер:",
        "lbl_smoothness": "Сглаживание:",
        "lbl_hardness": "Жесткость:",
        "btn_clear_mask": "🗑️ Очистить",
        "btn_apply_inpaint": "✨ Применить",
    },
    "zh": {
        "lang_name": "中文",
        "btn_open": "打开",
        "btn_save": "保存",
        "btn_compare": "对比",
        "btn_zoom_fit": "适应",
        "btn_kofi": "☕ 支持 Eliather",
        "tooltip_open": "打开图像 (Ctrl+O) 或 Ctrl+V",
        "tooltip_save": "保存 (Ctrl+S)",
        "tooltip_theme": "切换主题",
        "tooltip_lang": "更改语言",
        "tooltip_undo": "撤消 (Ctrl+Z)",
        "tooltip_redo": "重做 (Ctrl+Y)",
        "tooltip_reset": "恢复原始图像",
        "tooltip_zoom_out": "缩小",
        "tooltip_zoom_in": "放大",
        "tooltip_zoom_fit": "适应窗口",
        "tooltip_zoom_100": "缩放至 100%",
        "tooltip_compare": "对比",
        "tooltip_kofi": "支持开发",
        "tooltip_device": "切换 GPU / CPU",
        "status_ready": "准备就绪。拖动图像或点击“打开”。",
        "status_saved": "已保存到：{name}",
        "status_undo": "已撤消：{action}",
        "status_redo": "已重做：{action}",
        "status_reset": "已恢复原始图像。",
        "status_device_changed": "设备模式已更改为：{device}",
        "popup_save_success_title": "保存成功",
        "popup_save_success_msg": "已成功保存到：\n{path}",
        "popup_no_image_title": "无图像",
        "popup_no_image_msg": "没有图像可供保存。",
        "popup_error_save_title": "保存错误",
        "popup_error_save_msg": "无法保存：\n{error}",
        "popup_device_title": "设备已更新",
        "popup_device_msg": "处理设备：{device}",
        "popup_lang_changed_title": "语言已更改",
        "popup_lang_changed_msg": "请重新启动应用程序。",
        
        "brush_settings": "画笔设置",
        "lbl_size_thick": "粗细：",
        "lbl_opacity": "不透明度：",
        "header_batch": "批量处理：",
        "batch_desc": "自动从多张图像中去除背景。",
        "btn_batch_input": "📂 选择源文件夹...",
        "batch_no_folder": "未选择文件夹",
        "btn_batch_output": "📁 选择目标文件夹...",
        "lbl_batch_model": "使用的模型：",
        "chk_batch_clothing": "🛡️ 高级衣物检测器",
        "btn_run_batch": "🚀 开始批量处理",
        "batch_input_path": "源：{path}",
        "batch_output_path": "目标：{path}",
        
        "model_isnet-anime": "IS-Net Anime（适合动漫和插画）",
        "model_u2net_human_seg": "U2-Net Human（人物与衣物检测，保护白色衣服）",
        "model_birefnet-general": "BiRefNet（通用高清和复杂边缘）",
        "model_inspyrenet": "InSPyReNet Swin-B（真实摄影和肖像）",
        
        "scale_x2": "放大 2x",
        "scale_x4": "放大 4x（最大分辨率）",
        "scale_enhance_only": "仅锐化（不改变大小）",

        "btn_ok": "确定",
        
        "canvas_drop_title": "将图像拖至此处",
        "canvas_drop_subtitle": "或使用“打开”按钮 / 粘贴 (Ctrl+V)",
        "canvas_drop_formats": "支持 PNG, JPG, JPEG, WEBP 和 BMP",
        
        "sidebar_title": "工具",
        "sidebar_subtitle": "本地 AI 编辑和增强",
        "sidebar_tab_bg": "⬚\n去背景",
        "sidebar_tab_restore": "✦\n放大",
        "sidebar_tab_brush": "🖌\n画笔",
        "sidebar_tab_batch": "📂\n批量",
        "sidebar_shortcuts_col": "▸ 快捷键",
        "sidebar_shortcuts_exp": "▾ 快捷键",
        "shortcut_zoom": "鼠标滚轮",
        "shortcut_zoom_desc": "缩放",
        "shortcut_pan": "中键 / 空格",
        "shortcut_pan_desc": "平移",
        "shortcut_undo": "Ctrl+Z / Ctrl+Y",
        "shortcut_undo_desc": "撤消 / 重做",
        
        "header_model": "分割模型：",
        "model_desc": "去除背景，保留优化的透明度（适合动漫和插画）。",
        "chk_clothing": "🛡️ 高级衣物保护（防止白色衣服破洞）",
        "chk_clothing_tooltip": "结合语义检测，保护白色衬衫免受亮背景干扰。",
        "btn_remove_now": "✨ 立即去除背景",
        
        "header_crop": "画布调整：",
        "btn_autocrop_now": "⛶ 裁剪至内容",
        "btn_autocrop_tooltip": "请先应用去除背景",
        
        "header_bg": "背景：",
        "swatch_transparent": "透明",
        "swatch_white": "白色 (#FFFFFF)",
        "swatch_black": "黑色 (#000000)",
        "swatch_gray": "灰色 (#E2E8F0)",
        "swatch_custom": "自定义颜色...",
        "color_white": "白色 (#FFFFFF)",
        "color_black": "黑色 (#000000)",
        "color_gray": "灰色 (#E2E8F0)",
        
        "header_scale": "增强倍数：",
        "scale_desc": "使用 Real-ESRGAN 提高清晰度和分辨率。",
        "chk_tiling": "分块处理 (Tiling)",
        "chk_tiling_tooltip": "防止高分辨率图像的内存问题。",
        "btn_scale_now": "🚀 放大图像",
        
        "header_brush": "魔法画笔：",
        "brush_desc": "涂抹您想移除的对象，AI 将智能修复。",
        "btn_brush_mode": "🖌️ 画笔模式",
        "btn_paint": "🖌️ 涂抹",
        "btn_erase": "🧽 橡皮擦",
        "lbl_size": "大小：",
        "lbl_smoothness": "平滑：",
        "lbl_hardness": "硬度：",
        "btn_clear_mask": "🗑️ 清除涂抹",
        "btn_apply_inpaint": "✨ 应用 AI 填充",
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

_translator = Translator()

def tr(key, **kwargs):
    return _translator.get(key, **kwargs)

def get_current_language():
    return _translator.current_lang

def set_language(lang_code):
    _translator.set_language(lang_code)

def get_available_languages():
    return [(code, data["lang_name"]) for code, data in TRANSLATIONS.items()]
