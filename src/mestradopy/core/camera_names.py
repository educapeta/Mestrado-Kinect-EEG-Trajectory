"""NOMES das webcams, na ORDEM do indice do OpenCV (backend CAP_DSHOW).

Motivo (medido em 23/09/2026): o mapa indice->camera NAO e' estavel. No mesmo
dia, com as DUAS webcams externas ligadas, o OpenCV abria

    idx 0 = externa (vista ampla) | idx 1 = webcam do LAPTOP | idx 2 = externa (bancada)

e, quando as externas sumiram (hub USB fora), o idx 0 passou a ser a webcam do
LAPTOP -- e os programas que abrem "a camera auxiliar" passaram a mostrar o
rosto de quem esta' no laptop, sem dizer nada. NOME nao tem esse problema: a
webcam do laptop continua sendo "Integrated Camera" em qualquer ordem de porta.

Como o nome e' obtido: enumeracao DirectShow (CLSID_VideoInputDeviceCategory),
a MESMA categoria que o backend CAP_DSHOW do OpenCV enumera -- por isso a
posicao na lista E' o indice do OpenCV. Verificado nesta maquina: com uma unica
camera presente, a lista tem 1 nome, 'Integrated Camera', que e' exatamente a
camera que o OpenCV abre no indice 0.

Este modulo e' DELIBERADAMENTE leve (ctypes/comtypes apenas; sem cv2, numpy,
pykinect2), no mesmo espirito de `imu.py`: saber o nome de uma camera nao pode
exigir carregar o SDK do Kinect nem o MediaPipe. Toda funcao degrada em silencio
(lista vazia/None) se a enumeracao nao estiver disponivel -- nome de camera e'
informacao de diagnostico, nunca pre-requisito para o programa rodar.
"""
from __future__ import annotations

from ctypes import POINTER, c_ulong, c_void_p, c_wchar_p

from comtypes import (CLSCTX_INPROC_SERVER, COMMETHOD, GUID, HRESULT,
                      IUnknown, CoCreateInstance)

try:                                  # nome do dispositivo (VARIANT do COM)
    from comtypes.automation import VARIANT
except Exception:                     # pragma: no cover - comtypes incompleto
    VARIANT = None                    # type: ignore[assignment]

CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11D0-BD3B-00A0C911CE86}")
IID_IPropertyBag = GUID("{55272A00-42CB-11CE-8135-00AA004BB851}")

#: Como o nome da webcam do LAPTOP aparece no Windows. Comparado em minusculas e
#: por substring: "Integrated Camera", "Integrated Webcam", "HP TrueVision HD"
#: etc. NAO entra nenhuma camera externa (as da bancada sao "SIGMA-W780M" e
#: "Trust USB Camera" nesta montagem).
NOMES_DO_LAPTOP = ("integrated", "laptop", "notebook", "webcam do laptop")

_CACHE = None                         # type: list[str] | None


class IPropertyBag(IUnknown):
    """IPropertyBag (nome amigavel do dispositivo).

    O VARIANT e' declarado [out]: com [in, out] o DirectShow devolve
    E_INVALIDARG ('Parametro incorreto') e o nome nunca sai.
    """
    _iid_ = IID_IPropertyBag
    _methods_ = [
        COMMETHOD([], HRESULT, "Read", (["in"], c_wchar_p, "name"),
                  (["out"], POINTER(VARIANT), "var"),
                  (["in"], c_void_p, "errorlog")),
        COMMETHOD([], HRESULT, "Write", (["in"], c_wchar_p, "name"),
                  (["in"], POINTER(VARIANT), "var")),
    ]


class IMoniker(IUnknown):
    """IMoniker so' ate' BindToStorage.

    As posicoes da vtable tem de casar com IPersist + IPersistStream, que por
    isso aparecem declarados (GetClassID, IsDirtyStream, Load, Save,
    GetSizeMax) antes de BindToObject/BindToStorage.
    """
    _iid_ = GUID("{0000000F-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetClassID", (["out"], POINTER(GUID), "p")),
        COMMETHOD([], HRESULT, "IsDirtyStream"),
        COMMETHOD([], HRESULT, "Load", (["in"], c_void_p, "pStm")),
        COMMETHOD([], HRESULT, "Save", (["in"], c_void_p, "pStm"),
                  (["in"], c_void_p, "fClearDirty")),
        COMMETHOD([], HRESULT, "GetSizeMax", (["out"], POINTER(c_ulong), "p")),
        COMMETHOD([], HRESULT, "BindToObject", (["in"], c_void_p, "pbc"),
                  (["in"], c_void_p, "pmkToLeft"),
                  (["in"], POINTER(GUID), "riid"),
                  (["out"], POINTER(c_void_p), "ppv")),
        COMMETHOD([], HRESULT, "BindToStorage", (["in"], c_void_p, "pbc"),
                  (["in"], c_void_p, "pmkToLeft"),
                  (["in"], POINTER(GUID), "riid"),
                  (["out"], POINTER(POINTER(IPropertyBag)), "ppv")),
    ]


class IEnumMoniker(IUnknown):
    _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Next", (["in"], c_ulong, "celt"),
                  (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
                  (["out"], POINTER(c_ulong), "fetched")),
        COMMETHOD([], HRESULT, "Skip", (["in"], c_ulong, "celt")),
        COMMETHOD([], HRESULT, "Reset"),
    ]


class ICreateDevEnum(IUnknown):
    _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")
    _methods_ = [
        COMMETHOD([], HRESULT, "CreateClassEnumerator",
                  (["in"], POINTER(GUID), "clsid"),
                  (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnum"),
                  (["in"], c_ulong, "flags")),
    ]


#: Mensagem unica (fonte unica) para o erro mais caro desta bancada: usar a
#: webcam do laptop como auxiliar do Kinect. Ela fica no lugar do Kinect (sem
#: baseline) e, pior, o indice dela muda de um dia para o outro.
AVISO_LAPTOP = (
    "AVISO: a camera {indice} e' a webcam do LAPTOP ({nome}) e NAO serve como "
    "{papel}: fica no lugar do Kinect (sem baseline para a triangulacao) e o "
    "indice dela muda quando as webcams externas entram/saem do USB. Confira "
    "quem e' quem com `.venv\\Scripts\\python.exe tools\\diagnostico_cameras.py "
    "--salvar` e use o indice de uma camera EXTERNA."
)


def _nome_do_moniker(moniker):
    """FriendlyName do moniker, ou None (moniker sem property bag)."""
    try:
        bag = moniker.BindToStorage(None, None, IID_IPropertyBag)
        return str(bag.Read("FriendlyName", None))
    except Exception:                 # noqa: BLE001 - moniker sem property bag
        return None


def enumerar(recarregar=False):
    """Nomes das webcams na ORDEM do indice do OpenCV (CAP_DSHOW).

    Lista vazia = nao deu para enumerar (ou nao ha' camera nenhuma). O resultado
    fica em cache: a enumeracao mexe com COM/DirectShow e custa ~0,3 s, uma vez
    por processo basta.
    """
    global _CACHE
    if _CACHE is not None and not recarregar:
        return list(_CACHE)
    nomes = []
    if VARIANT is not None:
        try:
            dev_enum = CoCreateInstance(CLSID_SystemDeviceEnum, ICreateDevEnum,
                                        CLSCTX_INPROC_SERVER)
            enum_moniker = dev_enum.CreateClassEnumerator(
                CLSID_VideoInputDeviceCategory, 0)
            while enum_moniker is not None:
                lido = enum_moniker.Next(1)
                moniker, buscados = (lido if isinstance(lido, tuple)
                                     else (lido, None))
                # Fim da lista = S_FALSE, com pceltFetched = 0. Sem checar isso
                # o ponteiro anterior volta e o laco NAO termina.
                if moniker is None or not buscados:
                    break
                nome = _nome_do_moniker(moniker)
                nomes.append(nome if nome else "?")
        except Exception:             # noqa: BLE001 - sem COM: lista vazia
            nomes = []
    _CACHE = nomes
    return list(nomes)


def nomes_das_cameras(recarregar=False):
    """Lista de nomes; o indice do OpenCV e' a posicao na lista."""
    return enumerar(recarregar)


def mapa_de_cameras(recarregar=False):
    """{indice: nome} na ordem do OpenCV/CAP_DSHOW."""
    return dict(enumerate(enumerar(recarregar)))


def nome_da_camera(indice):
    """Nome da camera do indice (None se o indice nao existe)."""
    try:
        indice = int(indice)
    except (TypeError, ValueError):
        return None
    nomes = enumerar()
    if 0 <= indice < len(nomes):
        return nomes[indice]
    return None


def e_do_laptop(nome):
    """True se o nome for de uma webcam integrada de laptop/notebook."""
    if not nome:
        return False
    minusculo = str(nome).lower()
    return any(pista in minusculo for pista in NOMES_DO_LAPTOP)


def indices_fora_do_laptop():
    """Indices que NAO sao a webcam do laptop (candidatos a auxiliar)."""
    return [indice for indice, nome in mapa_de_cameras().items()
            if not e_do_laptop(nome)]


def aviso_laptop(indice, papel="webcam auxiliar"):
    """Texto de aviso se o indice for a webcam do laptop; senao None.

    Devolve texto (em vez de imprimir) para o chamador escolher o destino; a
    FRASE, porem, vive so' aqui.
    """
    nome = nome_da_camera(indice)
    if e_do_laptop(nome):
        return AVISO_LAPTOP.format(indice=indice, nome=nome, papel=papel)
    return None


def resumo(indices=None):
    """Linhas prontas para imprimir: 'idx N: nome [<- webcam do LAPTOP]'."""
    nomes = enumerar()
    if not nomes:
        return ["nenhum nome de webcam disponivel "
                "(enumeracao DirectShow vazia)"]
    escolhidos = (list(range(len(nomes))) if indices is None
                  else [int(i) for i in indices])
    linhas = []
    for indice in escolhidos:
        nome = nomes[indice] if 0 <= indice < len(nomes) else None
        if nome is None:
            linhas.append("idx %d: (nao existe)" % indice)
        else:
            marca = "  <- webcam do LAPTOP" if e_do_laptop(nome) else ""
            linhas.append("idx %d: %s%s" % (indice, nome, marca))
    return linhas
