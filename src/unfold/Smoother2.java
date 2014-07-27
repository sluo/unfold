package unfold;

import edu.mines.jtk.dsp.LocalSmoothingFilter;
import edu.mines.jtk.dsp.RecursiveExponentialFilter;
import static edu.mines.jtk.util.ArrayMath.*;

public class Smoother2 {

  public Smoother2(float sigma1, float sigma2, float[][] el) {
    _sigma1 = sigma1;
    _sigma2 = sigma2;
    _el = el;
  }

  public void apply(float[][] x1, float[][] x2) {
    apply(x1);
    apply(x2);
  }

  public void applyTranspose(float[][] x1, float[][] x2) {
    applyTranspose(x1);
    applyTranspose(x2);
  }

  public void apply(float[][][] x) {
    int n = x.length;
    for (int i=0; i<n; ++i) {
      smooth1(_sigma1,x[i]);
      smooth2(_sigma2,_el,x[i]);
      if (i<2) subtract(x[i]);
    }
  }

  public void applyTranspose(float[][][] x) {
    int n = x.length;
    for (int i=0; i<n; ++i) {
      if (i<2) subtract(x[i]);
      smooth2(_sigma2,_el,x[i]);
      smooth1(_sigma1,x[i]);
    }
  }

  public void apply(float[][] x) {
    smooth1(_sigma1,x);
    smooth2(_sigma2,_el,x);
    subtract(x);
  }

  public void applyTranspose(float[][] x) {
    subtract(x);
    smooth2(_sigma2,_el,x);
    smooth1(_sigma1,x);
  }

  // xt = a(u)
  public void apply(float[][] x1, float[][] x2, float[][] xt) {
    apply(x1);
    apply(x2);
    smooth1(_sigma1,xt);
    smooth2(_sigma2,_el,xt);
  }

  // xt = a(u)
  public void applyTranspose(float[][] x1, float[][] x2, float[][] xt) {
    applyTranspose(x1);
    applyTranspose(x2);
    smooth2(_sigma2,_el,xt);
    smooth1(_sigma1,xt);
  }
  

  ///////////////////////////////////////////////////////////////////////////
  // private

  private float _sigma1,_sigma2;
  private float[][] _el;

  private void subtract(float[][] x) {
    int n1 = x[0].length;
    int n2 = x.length;
    float xavg = sum(x)/(float)n1/(float)n2;
    sub(x,xavg,x);
  }

  private void subtract(float[] e, float[][] x) {
    int n1 = x[0].length;
    int n2 = x.length;
    double d = 0.0;
    for (int i2=0; i2<n2; ++i2)
      for (int i1=0; i1<n1; ++i1)
        d += e[i1]*x[i2][i1];
    float f = (float)d;
    for (int i2=0; i2<n2; ++i2)
      for (int i1=0; i1<n1; ++i1)
        x[i2][i1] -= f*e[i1];
  }

  // Smoothing for dimension 1.
  private void smooth1(float sigma, float[][] x) {
    new RecursiveExponentialFilter(sigma).apply1(x,x);
  }

  // Smoothing for dimension 2.
  private void smooth2(float sigma, float[][] s, float[][] x) {
    if (sigma<1.0f)
      return;
    float c = 0.5f*sigma*sigma;
    int n1 = x[0].length;
    int n2 = x.length;
    float[] st = fillfloat(1.0f,n2);
    float[] xt = zerofloat(n2);
    float[] yt = zerofloat(n2);
    LocalSmoothingFilter lsf = new LocalSmoothingFilter();
    for (int i1=0; i1<n1; ++i1) {
      if (s!=null) {
        for (int i2=0; i2<n2; ++i2)
          st[i2] = s[i2][i1];
      }
      for (int i2=0; i2<n2; ++i2)
        xt[i2] = x[i2][i1];
      lsf.apply(c,st,xt,yt);
      for (int i2=0; i2<n2; ++i2)
        x[i2][i1] = yt[i2];
    }
  }
}
