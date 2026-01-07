// Internationalization (i18n) Module
let currentLanguage = 'zh-TW';
let translations = {};

// Load translation file
async function loadTranslations(lang) {
    try {
        const response = await fetch(`locales/${lang}.json`);
        if (!response.ok) {
            throw new Error(`Failed to load ${lang}.json`);
        }
        translations = await response.json();
        return translations;
    } catch (error) {
        console.error(`Error loading translations for ${lang}:`, error);
        // Fallback to English if current language fails
        if (lang !== 'en') {
            return loadTranslations('en');
        }
        return {};
    }
}

// Translate a key (supports nested keys like "convert.title")
function t(key, params = {}) {
    const keys = key.split('.');
    let value = translations;
    
    for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
            value = value[k];
        } else {
            console.warn(`Translation key not found: ${key}`);
            return key;
        }
    }
    
    // Replace parameters in the translation string
    if (typeof value === 'string' && params) {
        return value.replace(/\{(\w+)\}/g, (match, param) => {
            return params[param] !== undefined ? params[param] : match;
        });
    }
    
    return value || key;
}

// Update all elements with data-i18n attribute
function updateTranslations() {
    // Update elements with data-i18n
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = t(key);
        
        if (element.tagName === 'INPUT' && element.type === 'text' || element.tagName === 'INPUT' && element.type === 'number') {
            // For input placeholders
            if (element.hasAttribute('data-i18n-placeholder')) {
                element.placeholder = translation;
            }
        } else if (element.tagName === 'OPTION') {
            // For option elements, update text content
            element.textContent = translation;
        } else {
            // For other elements, update text content but preserve HTML structure
            if (element.children.length === 0) {
                element.textContent = translation;
            } else {
                // If element has children, only update if it's a simple text node
                const textNodes = Array.from(element.childNodes).filter(node => node.nodeType === 3);
                if (textNodes.length > 0) {
                    textNodes[0].textContent = translation;
                }
            }
        }
    });
    
    // Update select options that have data-i18n
    document.querySelectorAll('select option[data-i18n]').forEach(option => {
        const key = option.getAttribute('data-i18n');
        option.textContent = t(key);
    });
}

// Change language
async function changeLanguage(lang) {
    currentLanguage = lang;
    await loadTranslations(lang);
    updateTranslations();
    
    // Update HTML lang attribute
    document.documentElement.lang = lang;
    
    // Save language preference
    localStorage.setItem('preferredLanguage', lang);
    
    // Update language selector
    const langSelect = document.getElementById('language-select');
    if (langSelect) {
        langSelect.value = lang;
    }
}

// Initialize i18n
async function initI18n() {
    // Get saved language preference or default to zh-TW
    const savedLang = localStorage.getItem('preferredLanguage') || 'zh-TW';
    await changeLanguage(savedLang);
    
    // Set up language selector
    const langSelect = document.getElementById('language-select');
    if (langSelect) {
        langSelect.value = savedLang;
        langSelect.addEventListener('change', (e) => {
            changeLanguage(e.target.value);
        });
    }
}

// Export functions for use in other scripts
window.i18n = {
    t,
    changeLanguage,
    initI18n,
    updateTranslations,
    getCurrentLanguage: () => currentLanguage
};

