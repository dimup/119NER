# 각 태그 토큰 대한 출력 일정화
def rule_based(inf_list):
    loc_list = []
    org_list = []
    noh_list = []
    emr_list = []

    for obj in inf_list:
        if obj[0] == 'LOC':
            loc_result = loc_rule(obj[1])
            loc_list.append(loc_result)
        elif obj[0] == 'ORG':
            org_result = org_rule(obj[1])
            org_list.append(org_result)
        elif obj[0] == 'NOH':
            noh_result = noh_rule(obj[1])
            noh_list.append(noh_result)
        elif obj[0] == 'EMR':
            emr_result = emr_rule(obj[1])
            emr_list.append(emr_result)
        else:
            pass

    loc = make_string(loc_list)
    org = make_string(org_list)
    noh = make_string(noh_list)
    emr = make_string(emr_list)

    return loc, org, noh, emr


# EMR tag
def emr_rule(emr):
    if '불' in str(emr):
        return "화재"
    elif "연기" in str(emr):
        return "화재 의심"
    elif "쓰러" in str(emr):
        return "사람 쓰러짐"
    else:
        return None


# LOC tag
def loc_rule(loc):
    if "에서" in str(loc):
        loc_after = str(loc).replace("에서", '')
        return loc_after
    else:
        return str(loc)


# ORG tag
def org_rule(org):
    return str(org)


# NOH tag
def noh_rule(noh):
    if str(noh)[-1] == '이':
        noh = str(noh).replace('이', '')
    else:
        noh = str(noh)
    if '명' in noh or '분' in noh:
        if '분' in noh:
            noh_result = noh.replace('분', ' 명')
            return noh_result
        else:
            noh_result = noh.replace('명', ' 명')
            return noh_result
    else:
        return None


# 추출된 결과가 없다면, 단순 공백이 아닌 "정보 없음"으로 출력되도록 함
def make_string(target_list):
    if not target_list:
        return "정보 없음"
    elif None in target_list:
        filtered_target_list = list(filter(None, target_list))
        filtered_result = ", ".join(filtered_target_list)
        if not filtered_result:
            return "정보 없음"
        else:
            return filtered_result
    else:
        return ", ".join(target_list)


# SER tag (규칙 기반 모델에서만 취급하는 태그)
def ser_rule(text):
    ser_list = ["엄청", "크게", "큰", "심하게", "많이", "완전",
                "심히", "빨리", "서둘러", "급해요", "코로나"]
    for target in ser_list:
        if target in text:
            return "긴급"

    return "일반"
