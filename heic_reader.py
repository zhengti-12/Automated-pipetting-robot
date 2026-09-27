"""
heic_reader.py — read iPhone .HEIC photos into an OpenCV BGR array.

Tries pillow-heif first (pip install pillow-heif). If that isn't installed,
falls back to a minimal HEIF parser that decodes the HEVC tiles with ffmpeg
(needs `ffmpeg` on your PATH).
"""
import struct, subprocess
import numpy as np, cv2


def read_heic(path):
    try:
        import pillow_heif
        h = pillow_heif.open_heif(path)
        return cv2.cvtColor(np.asarray(h), cv2.COLOR_RGB2BGR)
    except ImportError:
        return _read_heic_ffmpeg(path)


def _read_heic_ffmpeg(path):
    data = open(path, 'rb').read()
    def boxes(buf,off,end):
        while off<end:
            size,typ=struct.unpack('>I4s',buf[off:off+8]); hdr=8
            if size==1: size=struct.unpack('>Q',buf[off+8:off+16])[0]; hdr=16
            if size==0: size=end-off
            yield typ.decode('latin1'),off+hdr,off+size; off+=size
    top={t:(s,e) for t,s,e in boxes(data,0,len(data))}
    ms,me=top['meta']; ms+=4
    meta={t:(s,e) for t,s,e in boxes(data,ms,me)}
    def rd(b,o,n): return int.from_bytes(b[o:o+n],'big') if n else 0
    # pitm
    s,_=meta['pitm']; v=data[s]; primary=rd(data,s+4,2 if v==0 else 4)
    # iinf
    s,e=meta['iinf']; v=data[s]; o=s+4; cnt=rd(data,o,2 if v==0 else 4); o+=2 if v==0 else 4
    itypes={}
    for t,bs,be in boxes(data,o,e):
        v=data[bs]
        if v>=2:
            iid=rd(data,bs+4,2 if v==2 else 4); p=bs+4+(2 if v==2 else 4)+2
            itypes[iid]=data[p:p+4].decode()
    # iloc
    s,e=meta['iloc']; v=data[s]; o=s+4
    a=data[o]; b=data[o+1]; os_,ls,bos=a>>4,a&15,b>>4; isz=(b&15) if v in(1,2) else 0; o+=2
    n=rd(data,o,2 if v<2 else 4); o+=2 if v<2 else 4
    loc={}
    for _ in range(n):
        iid=rd(data,o,2 if v<2 else 4); o+=2 if v<2 else 4
        cm=rd(data,o+1,1)&15 if v in(1,2) else 0
        if v in(1,2): o+=2
        o+=2; base=rd(data,o,bos); o+=bos
        ec=rd(data,o,2); o+=2; ext=[]
        for _ in range(ec):
            if v in(1,2) and isz: o+=isz
            eo=rd(data,o,os_); o+=os_; el=rd(data,o,ls); o+=ls; ext.append(((meta['idat'][0] if cm==1 else 0)+base+eo,el))
        loc[iid]=ext
    def item(i): return b''.join(data[a:a+l] for a,l in loc[i])
    # iref
    s,e=meta['iref']; v=data[s]; w=2 if v==0 else 4; refs={}
    for t,bs,be in boxes(data,s+4,e):
        fr=rd(data,bs,w); c=rd(data,bs+w,2); refs.setdefault((t,fr),[rd(data,bs+w+2+w*k,w) for k in range(c)])
    # iprp
    s,e=meta['iprp']; ip={t:(a,b) for t,a,b in boxes(data,s,e)}
    props=[(t,a,b) for t,a,b in boxes(data,*ip['ipco'])]
    s,e=ip['ipma']; v=data[s]; fl=rd(data,s+1,3); o=s+4; n=rd(data,o,4); o+=4; assoc={}
    for _ in range(n):
        iid=rd(data,o,2 if v<1 else 4); o+=2 if v<1 else 4; c=data[o]; o+=1; L=[]
        for _ in range(c):
            if fl&1: x=rd(data,o,2); o+=2; L.append(x&0x7fff)
            else: x=data[o]; o+=1; L.append(x&0x7f)
        assoc[iid]=L
    def getprop(iid,t):
        for idx in assoc.get(iid,[]):
            pt,a,b=props[idx-1]
            if pt==t: return a,b
    def hvcc_nals(iid):
        a,b=getprop(iid,'hvcC'); p=a+22; na=data[p]; p+=1; out=b''
        for _ in range(na):
            p+=1; nn=rd(data,p,2); p+=2
            for _ in range(nn):
                l=rd(data,p,2); p+=2; out+=b'\0\0\0\1'+data[p:p+l]; p+=l
        return out
    def decode_tile(iid):
        raw=item(iid); out=hvcc_nals(iid); p=0
        while p<len(raw):
            l=rd(raw,p,4); out+=b'\0\0\0\1'+raw[p+4:p+4+l]; p+=4+l
        r=subprocess.run(['ffmpeg','-loglevel','error','-f','hevc','-i','-','-frames:v','1','-f','image2pipe','-vcodec','png','-'],input=out,capture_output=True)
        return cv2.imdecode(np.frombuffer(r.stdout,np.uint8),1)
    if itypes[primary]=='grid':
        g=item(primary); fl=g[1]; rows,cols=g[2]+1,g[3]+1; w=4 if fl&1 else 2
        W=rd(g,4,w); H=rd(g,4+w,w)
        tiles=[decode_tile(t) for t in refs[('dimg',primary)]]
        th,tw=tiles[0].shape[:2]
        canvas=np.zeros((rows*th,cols*tw,3),np.uint8)
        for k,t in enumerate(tiles): r,c=divmod(k,cols); canvas[r*th:(r+1)*th,c*tw:(c+1)*tw]=t
        img=canvas[:H,:W]
    else: img=decode_tile(primary)
    ir=getprop(primary,'irot')
    if ir:
        ang=data[ir[0]]&3
        for _ in range(ang): img=cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


if __name__ == "__main__":
    import sys
    im = read_heic(sys.argv[1]); cv2.imwrite(sys.argv[2], im); print(im.shape)
