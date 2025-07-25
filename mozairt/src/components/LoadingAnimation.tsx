// src/components/LoadingAnimation.tsx
import { motion } from 'framer-motion';

// 부모 컨테이너 variant (수정 없음)
const containerVariants = {
  animate: {
    transition: {
      staggerChildren: 0.15,
    },
  },
};

// 각 점에 대한 variant (수정 없음)
const dotVariants = {
  animate: {
    y: [0, -20, 0],
    height: ['0.75rem', '1.7rem', '0.75rem'],
    borderRadius: ['50%', '30%', '50%'],
    transition: {
      duration: 1.5,
      ease: 'easeInOut',
      repeat: Infinity,
    },
  },
};

const LoadingAnimation = () => {
  return (
    // 1. 부모 컨테이너에서 잘못된 initial="initial" 속성 제거
    <motion.div
      variants={containerVariants}
      animate="animate"
      className="flex justify-center items-end gap-2 h-10"
    >
      {/* 2. 각 점에 기본 크기를 지정해주는 w-3 h-3 클래스 추가 */}
      <motion.div variants={dotVariants} className="w-2 h-3 bg-gray-400" />
      <motion.div variants={dotVariants} className="w-2 h-3 bg-gray-400" />
      <motion.div variants={dotVariants} className="w-2 h-3 bg-gray-400" />
    </motion.div>
  );
};

export default LoadingAnimation;