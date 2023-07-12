import string
from soynlp.hangle import compose



def construct_vocab():
        """ Construct vocab using possible Korean characters and special tokens """
        
        # Add special tokens
        special_tokens = ['<PAD>', '<SOW>', '<EOW>', '<UNK>']

        # Add possible Korean characters
        KP = KoreanPhoneme()
        korean_chars = [compose(c1, v, c2)
                        for c1 in KP.chosung_list
                        for v in KP.jungsung_list
                        for c2 in KP.jongsung_list]
        
        # Add special characters
        special_chars = [s for s in string.printable]

        chardict = {char: i
                    for i, char in enumerate(special_tokens + korean_chars + special_chars)}
        return chardict

    
    
class KoreanPhoneme():
    """ Features of Korean phonemes, especially called jamos which mean consonants and vowels """
    
    def __init__(self):
        # a single Korean character is composed with chosung, jungsung, and jongsung
        self.chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ',
                             'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
                             'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ',
                             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ',
                              'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
                              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
                              'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        self.jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ',
                              'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
                              'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ',
                              'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                              'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ',
                              'ㅌ', 'ㅍ', 'ㅎ']
        
        # each of chosung, jungsung and jongsong is classfied to jaum(consonant) or moum(vowel)
        self.jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ',
                          'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ',
                          'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ',
                          'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ',
                          'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ',
                          'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
        self.moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ',
                          'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
                          'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
                          'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
        
        # near jamos on the keyboard which can occur with unintentional typos
        # ㅂ, ㅈ, ㄷ, ㄱ, ㅅ
        self.bieup_near  = ['ㅈ', 'ㅁ']
        self.jieut_near  = ['ㅂ', 'ㄷ', 'ㄴ']
        self.digeut_near = ['ㅈ', 'ㄱ', 'ㅇ'] 
        self.giyeok_near = ['ㄷ', 'ㅅ', 'ㄹ']
        self.siot_near   = ['ㄱ', 'ㅎ']
        # ㅁ, ㄴ, ㅇ, ㄹ, ㅎ
        self.mieum_near  = ['ㅂ', 'ㄴ', 'ㅋ']
        self.nieun_near  = ['ㅈ', 'ㅁ', 'ㅇ', 'ㅌ']
        self.ieung_near  = ['ㄷ', 'ㄴ', 'ㄹ', 'ㅊ']
        self.rieul_near  = ['ㄱ', 'ㅇ', 'ㅎ', 'ㅍ']
        self.hieut_near  = ['ㅅ', 'ㄹ', 'ㅍ']
        # ㅋ, ㅌ, ㅊ, ㅍ, 
        self.kieuk_near  = ['ㅁ', 'ㅌ']
        self.tieut_near  = ['ㄴ', 'ㅋ', 'ㅊ']
        self.chieut_near = ['ㅇ', 'ㅌ', 'ㅍ']
        self.pieup_near  = ['ㄹ', 'ㅊ']
        
        # ㅛ, ㅕ, ㅑ, ㅐ, ㅔ
        self.yo_near  = ['ㅕ', 'ㅗ']
        self.yeo_near = ['ㅛ', 'ㅑ', 'ㅓ']
        self.ya_near  = ['ㅕ', 'ㅐ', 'ㅓ']
        self.ae_near  = ['ㅑ', 'ㅔ', 'ㅏ']
        self.e_near   = ['ㅐ', 'ㅣ']
        # ㅗ, ㅓ, ㅏ, ㅣ
        self.o_near   = ['ㅛ', 'ㅓ', 'ㅠ']
        self.eo_near  = ['ㅕ', 'ㅗ', 'ㅏ', 'ㅜ']
        self.a_near   = ['ㅑ', 'ㅓ', 'ㅣ', 'ㅡ']
        self.i_near   = ['ㅔ', 'ㅏ']
        # ㅠ, ㅜ, ㅡ
        self.yu_near  = ['ㅗ', 'ㅜ']
        self.u_near   = ['ㅓ', 'ㅠ', 'ㅡ']
        self.eu_near  = ['ㅏ', 'ㅜ']
        
        # ㄲ, ㄸ, ㅃ, ㅆ, ㅉ
        self.ssang_giyeok_near = ['ㄱ'] + self.giyeok_near
        self.ssang_digeut_near = ['ㄷ'] + self.digeut_near
        self.ssang_bieup_near  = ['ㅂ'] + self.bieup_near
        self.ssang_siot_near   = ['ㅅ'] + self.siot_near
        self.ssang_jieut_near  = ['ㅈ'] + self.jieut_near
        
        # ㄳ, ㄵ, ㄶ, ㄺ, ㄻ, ㄼ, ㄾ, ㄿ, ㅀ, ㅄ
        self.giyeok_siot_near  = ['ㄱ', 'ㅅ']
        self.nieun_jieut_near  = ['ㄴ', 'ㅈ']
        self.nieun_hieut_near  = ['ㄴ', 'ㅎ']
        self.rieul_giyeok_near = ['ㄹ', 'ㄱ']
        self.rieul_mieum_near  = ['ㄹ', 'ㅁ']
        self.rieul_bieup_near  = ['ㄹ', 'ㅂ']
        self.rieul_tieut_near  = ['ㄹ', 'ㅌ']
        self.rieul_pieup_near  = ['ㄹ', 'ㅍ'] 
        self.rieul_hieut_near  = ['ㄹ', 'ㅎ']
        self.bieup_siot_near   = ['ㅂ', 'ㅅ']
        
        # ㅒ, ㅖ, ㅚ, ㅟ, ㅢ
        self.yae_near = ['ㅐ'] + self.ae_near
        self.ye_near  = ['ㅔ'] + self.e_near
        self.oe_near  = ['ㅗ', 'ㅣ']
        self.wi_near  = ['ㅜ', 'ㅣ']
        self.ui_near  = ['ㅡ', 'ㅣ']
        # ㅘ, ㅝ, ㅙ, ㅞ
        self.wa_near  = ['ㅗ', 'ㅏ']
        self.wo_near  = ['ㅜ', 'ㅓ']
        self.wae_near = ['ㅗ', 'ㅐ']
        self.we_near  = ['ㅜ', 'ㅔ']
        
        
    def return_near_jamos(self, jamo):
        """ Return near jamos on the keyboard """
              
        if jamo == ' ': return self.jongsung_list
        
        elif jamo == 'ㅂ': return self.bieup_near
        elif jamo == 'ㅈ': return self.jieut_near
        elif jamo == 'ㄷ': return self.digeut_near
        elif jamo == 'ㄱ': return self.giyeok_near
        elif jamo == 'ㅅ': return self.siot_near
        elif jamo == 'ㅁ': return self.mieum_near
        elif jamo == 'ㄴ': return self.nieun_near
        elif jamo == 'ㅇ': return self.ieung_near
        elif jamo == 'ㄹ': return self.rieul_near
        elif jamo == 'ㅎ': return self.hieut_near
        elif jamo == 'ㅋ': return self.kieuk_near
        elif jamo == 'ㅌ': return self.tieut_near
        elif jamo == 'ㅊ': return self.chieut_near
        elif jamo == 'ㅍ': return self.pieup_near
       
        elif jamo == 'ㅛ': return self.yo_near
        elif jamo == 'ㅕ': return self.yeo_near
        elif jamo == 'ㅑ': return self.ya_near
        elif jamo == 'ㅐ': return self.ae_near
        elif jamo == 'ㅔ': return self.e_near
        elif jamo == 'ㅗ': return self.o_near
        elif jamo == 'ㅓ': return self.eo_near
        elif jamo == 'ㅏ': return self.a_near
        elif jamo == 'ㅣ': return self.i_near
        elif jamo == 'ㅠ': return self.yu_near
        elif jamo == 'ㅜ': return self.u_near
        elif jamo == 'ㅡ': return self.eu_near
      
        elif jamo == 'ㄲ': return self.ssang_giyeok_near
        elif jamo == 'ㄸ': return self.ssang_digeut_near
        elif jamo == 'ㅃ': return self.ssang_bieup_near
        elif jamo == 'ㅆ': return self.ssang_siot_near
        elif jamo == 'ㅉ': return self.ssang_jieut_near
        
        elif jamo == 'ㄳ': return self.giyeok_siot_near
        elif jamo == 'ㄵ': return self.nieun_jieut_near
        elif jamo == 'ㄶ': return self.nieun_hieut_near
        elif jamo == 'ㄺ': return self.rieul_giyeok_near
        elif jamo == 'ㄻ': return self.rieul_mieum_near
        elif jamo == 'ㄼ': return self.rieul_bieup_near
        elif jamo == 'ㄾ': return self.rieul_tieut_near
        elif jamo == 'ㄿ': return self.rieul_pieup_near
        elif jamo == 'ㅀ': return self.rieul_hieut_near
        elif jamo == 'ㅄ': return self.bieup_siot_near
        
        elif jamo == 'ㅒ': return self.yae_near
        elif jamo == 'ㅖ': return self.ye_near
        elif jamo == 'ㅚ': return self.oe_near
        elif jamo == 'ㅟ': return self.wi_near
        elif jamo == 'ㅢ': return self.ui_near
        elif jamo == 'ㅘ': return self.wa_near
        elif jamo == 'ㅝ': return self.wo_near
        elif jamo == 'ㅙ': return self.wae_near
        elif jamo == 'ㅞ': return self.we_near