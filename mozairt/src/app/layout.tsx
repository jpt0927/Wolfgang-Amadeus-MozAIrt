import type { Metadata } from "next";
import "./globals.css";
import ClickFX from "@/components/ClickFX";

export const metadata: Metadata = {
  title: "AI Music Studio",
  description: "AI로 만드는 나만의 음악",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <head>
        {/* Pretendard 폰트 CDN 추가 */}
        <link
          rel="stylesheet"
          as="style"
          crossOrigin="anonymous"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css"
        />
      </head>
      <body>
        <div id="space-background">
          <ClickFX />
        </div>
        {children}
        </body>
    </html>
  );
}